"""
Doosan M1013 - SAC + HER 학습 스크립트 (Torque 제어 전용)

설계:
  - Observation: qpos(6) + qvel(6) + ee_pose(7) + ee_vel(7) + desired_speed(1) = 27D
  - Goal: ee_pos(3) + ee_quat(4) = 7D
  - Obstacles: 10 × 7D = 70D (별도 MLP 인코더 처리)
  - Reward: -(pos_dist + 0.3 × angle) + 성공 보너스(+10) + 충돌 패널티(-5)
  - VecNormalize: obs 정규화 ON, reward 정규화 OFF
  - CustomExtractor: ObstacleAwareExtractor (robot MLP + obstacle MLP → concat)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import robot_env  # M1013Reach-v0 등록
from robot_env import (M1013Env as RobotArmEnv, MAX_OBSTACLES,
                       OBS_UNLOCK_THRESHOLD, SPEED_UNLOCK_THRESHOLD,
                       SPEED_WEIGHT_MAX, SPEED_WEIGHT_STEP)

LOG_DIR_BASE   = "./logs"
MODEL_DIR_BASE = "./models/torque"


# ── Custom Feature Extractor ──────────────────────────────────────────────

class ObstacleAwareExtractor(BaseFeaturesExtractor):
    """
    장애물 정보를 별도 MLP로 처리한 뒤 로봇 상태와 concat.

    구조:
      robot_obs(27) + achieved_goal(7) + desired_goal(7) = 41D → robot_net → 256D
      obstacles(70D) → obstacle_net → 64D
      concat → 320D (features_dim)
    """

    def __init__(self, observation_space: spaces.Dict,
                 robot_out: int = 256, obstacle_out: int = 64):
        robot_dim = (
            observation_space["observation"].shape[0]    # 27
            + observation_space["achieved_goal"].shape[0]  # 7
            + observation_space["desired_goal"].shape[0]   # 7
        )  # = 41
        obstacle_dim = observation_space["obstacles"].shape[0]  # 70
        features_dim = robot_out + obstacle_out  # 320

        super().__init__(observation_space, features_dim)

        self.robot_net = nn.Sequential(
            nn.Linear(robot_dim, 256),
            nn.ReLU(),
            nn.Linear(256, robot_out),
            nn.ReLU(),
        )
        self.obstacle_net = nn.Sequential(
            nn.Linear(obstacle_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obstacle_out),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        robot_feat = torch.cat([
            observations["observation"],
            observations["achieved_goal"],
            observations["desired_goal"],
        ], dim=1)
        return torch.cat([
            self.robot_net(robot_feat),
            self.obstacle_net(observations["obstacles"]),
        ], dim=1)


# ── Callbacks ────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    성공률 기반 Curriculum 콜백.
    pos/ori threshold, init_range, max_obs_count, speed_weight 자동 조정.

    성공률 >= 70%:
      - pos/ori threshold × 0.8,  init_range × 1.5
      - pos_threshold < OBS_UNLOCK_THRESHOLD  → max_obs_count + 1
      - pos_threshold < SPEED_UNLOCK_THRESHOLD → speed_weight += SPEED_WEIGHT_STEP

    성공률 < 20%:
      - pos/ori threshold × 1.2  (나머지 유지)
    """

    def __init__(self, print_freq=500,
                 init_threshold=0.30, init_ori_threshold=None,
                 init_range=None, init_max_obs_count=0, init_speed_weight=0.0,
                 threshold_save_path=None, ori_threshold_save_path=None,
                 init_range_save_path=None, max_obs_count_save_path=None,
                 speed_weight_save_path=None,
                 eval_vec_env=None,
                 warmup_episodes=0,
                 verbose=0):
        super().__init__(verbose)
        self.successes       = []
        self.episodes        = 0
        self._last_logged_ep = 0
        self.print_freq      = print_freq
        self.warmup_episodes = warmup_episodes

        self.success_threshold = init_threshold
        self.ori_threshold     = (init_ori_threshold
                                  if init_ori_threshold is not None
                                  else RobotArmEnv.ORI_THRESHOLD_INIT)
        self.init_range        = (init_range
                                  if init_range is not None
                                  else RobotArmEnv.INIT_RANGE_INIT)
        self.max_obs_count     = int(init_max_obs_count)
        self.speed_weight      = float(init_speed_weight)

        self.threshold_save_path      = threshold_save_path
        self.ori_threshold_save_path  = ori_threshold_save_path
        self.init_range_save_path     = init_range_save_path
        self.max_obs_count_save_path  = max_obs_count_save_path
        self.speed_weight_save_path   = speed_weight_save_path
        self.eval_vec_env             = eval_vec_env

    def _unwrap_envs(self, vec_env):
        envs = vec_env.envs if hasattr(vec_env, "envs") else vec_env.venv.envs
        result = []
        for env in envs:
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            result.append(unwrapped)
        return result

    def _set_curriculum(self, threshold: float, ori_threshold: float,
                        init_range: float = None, max_obs_count: int = None,
                        speed_weight: float = None):
        self.success_threshold = float(np.clip(
            threshold, RobotArmEnv.SUCCESS_THRESHOLD_MIN, RobotArmEnv.SUCCESS_THRESHOLD_MAX,
        ))
        self.ori_threshold = float(np.clip(
            ori_threshold, RobotArmEnv.ORI_THRESHOLD_MIN, RobotArmEnv.ORI_THRESHOLD_MAX,
        ))
        if init_range is not None:
            self.init_range = float(np.clip(
                init_range, RobotArmEnv.INIT_RANGE_INIT, RobotArmEnv.INIT_RANGE_MAX,
            ))
        if max_obs_count is not None:
            self.max_obs_count = int(np.clip(max_obs_count, 0, MAX_OBSTACLES))
        if speed_weight is not None:
            self.speed_weight = float(np.clip(speed_weight, 0.0, SPEED_WEIGHT_MAX))

        all_envs = self._unwrap_envs(self.training_env)
        if self.eval_vec_env is not None:
            all_envs += self._unwrap_envs(self.eval_vec_env)

        for env in all_envs:
            env.set_success_threshold(self.success_threshold, self.ori_threshold)
            env.set_init_range(self.init_range)
            env.set_max_obs_count(self.max_obs_count)
            env.set_speed_weight(self.speed_weight)

        if self.threshold_save_path:
            with open(self.threshold_save_path, "w") as f:
                f.write(str(self.success_threshold))
        if self.ori_threshold_save_path:
            with open(self.ori_threshold_save_path, "w") as f:
                f.write(str(self.ori_threshold))
        if self.init_range_save_path:
            with open(self.init_range_save_path, "w") as f:
                f.write(str(self.init_range))
        if self.max_obs_count_save_path:
            with open(self.max_obs_count_save_path, "w") as f:
                f.write(str(self.max_obs_count))
        if self.speed_weight_save_path:
            with open(self.speed_weight_save_path, "w") as f:
                f.write(str(self.speed_weight))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.successes.append(float(info.get("success", False)))
                self.episodes += 1

        if self.episodes < self._last_logged_ep + self.print_freq:
            return True

        recent = self.successes[-self.print_freq:]
        rate   = np.mean(recent) * 100

        if self.episodes < self.warmup_episodes:
            print(
                f"  [Ep {self.episodes:>7d}] 성공률: {rate:5.1f}% | "
                f"[warmup 중 — {self.warmup_episodes - self.episodes}ep 남음]"
            )
            self._last_logged_ep = self.episodes
            self.logger.record("train/success_rate", rate)
            return True

        if rate >= 70.0:
            new_max_obs    = self.max_obs_count
            new_speed_w    = self.speed_weight
            if (self.success_threshold <= OBS_UNLOCK_THRESHOLD
                    and self.max_obs_count < MAX_OBSTACLES):
                new_max_obs = self.max_obs_count + 1
            if (self.success_threshold <= SPEED_UNLOCK_THRESHOLD
                    and self.speed_weight < SPEED_WEIGHT_MAX):
                new_speed_w = self.speed_weight + SPEED_WEIGHT_STEP

            self._set_curriculum(
                self.success_threshold * 0.8,
                self.ori_threshold * 0.8,
                init_range=self.init_range * 1.5,
                max_obs_count=new_max_obs,
                speed_weight=new_speed_w,
            )
        elif rate < 20.0:
            self._set_curriculum(
                self.success_threshold * 1.2,
                self.ori_threshold * 1.2,
            )

        print(
            f"  [Ep {self.episodes:>7d}] "
            f"성공률: {rate:5.1f}% | "
            f"pos: {self.success_threshold*100:.2f}cm | "
            f"ori: {np.degrees(self.ori_threshold):.1f}° | "
            f"init: ±{np.degrees(self.init_range):.0f}° | "
            f"obs: {self.max_obs_count}/{MAX_OBSTACLES} | "
            f"spd_w: {self.speed_weight:.2f} | "
            f"ent: {float(self.model.target_entropy):.2f}"
        )
        self._last_logged_ep = self.episodes
        self.logger.record("train/success_rate",        rate)
        self.logger.record("train/success_threshold",   self.success_threshold)
        self.logger.record("train/ori_threshold_deg",   np.degrees(self.ori_threshold))
        self.logger.record("train/init_range_deg",      np.degrees(self.init_range))
        self.logger.record("train/max_obs_count",       self.max_obs_count)
        self.logger.record("train/speed_weight",        self.speed_weight)
        self.logger.record("train/target_entropy",      float(self.model.target_entropy))
        return True


class EntCoefFloorCallback(BaseCallback):
    """ent_coef가 설정값 아래로 떨어지지 않도록 log_ent_coef를 매 스텝 클램핑."""

    def __init__(self, min_ent_coef: float = 0.03, verbose=0):
        super().__init__(verbose)
        self.min_log_ent_coef = math.log(min_ent_coef)

    def _on_step(self) -> bool:
        self.model.log_ent_coef.data.clamp_(min=self.min_log_ent_coef)
        return True


class VecNormSaveCallback(BaseCallback):
    """VecNormalize 통계를 주기적으로 저장."""

    def __init__(self, save_path: str, save_freq: int = 10_000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(self.save_path)
        return True


# ── 학습 함수 ────────────────────────────────────────────────────────────

def make_env():
    env = gym.make("M1013Reach-v0")
    return Monitor(env)


def train(total_timesteps: int = 2_000_000, n_envs: int = 8, resume: str = None):
    log_dir   = LOG_DIR_BASE
    model_dir = MODEL_DIR_BASE
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 65)
    print("  Doosan M1013 — SAC + HER (Torque + Dynamic Obstacles)")
    print("=" * 65)

    vec_env_raw  = make_vec_env(make_env, n_envs=n_envs)
    eval_env_raw = make_vec_env(make_env, n_envs=1)

    vecnorm_path             = os.path.join(model_dir, "vecnormalize.pkl")
    threshold_save_path      = os.path.join(model_dir, "success_threshold.txt")
    ori_threshold_save_path  = os.path.join(model_dir, "ori_threshold.txt")
    init_range_save_path     = os.path.join(model_dir, "init_range.txt")
    max_obs_count_save_path  = os.path.join(model_dir, "max_obs_count.txt")
    speed_weight_save_path   = os.path.join(model_dir, "speed_weight.txt")

    if resume and os.path.exists(vecnorm_path):
        vec_env  = VecNormalize.load(vecnorm_path, vec_env_raw)
        vec_env.training    = True
        vec_env.norm_reward = False
        eval_env = VecNormalize.load(vecnorm_path, eval_env_raw)
        eval_env.training   = False
        eval_env.norm_reward = False
        print(f"  VecNormalize 복원: {vecnorm_path}")
    else:
        vec_env  = VecNormalize(vec_env_raw,  norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False,
                                training=False, clip_obs=10.0)

    # Curriculum 상태 복원
    init_threshold = RobotArmEnv.SUCCESS_THRESHOLD_INIT
    if resume and os.path.exists(threshold_save_path):
        with open(threshold_save_path) as f:
            init_threshold = float(f.read().strip())
        print(f"  pos threshold 복원: {init_threshold*100:.2f}cm")

    init_ori_threshold = RobotArmEnv.ORI_THRESHOLD_INIT
    if resume and os.path.exists(ori_threshold_save_path):
        with open(ori_threshold_save_path) as f:
            init_ori_threshold = float(f.read().strip())
        print(f"  ori threshold 복원: {np.degrees(init_ori_threshold):.1f}°")

    init_range = RobotArmEnv.INIT_RANGE_INIT
    if resume and os.path.exists(init_range_save_path):
        with open(init_range_save_path) as f:
            init_range = float(f.read().strip())
        print(f"  init range 복원: ±{np.degrees(init_range):.0f}°")

    init_max_obs_count = 0
    if resume and os.path.exists(max_obs_count_save_path):
        with open(max_obs_count_save_path) as f:
            init_max_obs_count = int(f.read().strip())
        print(f"  max_obs_count 복원: {init_max_obs_count}")

    init_speed_weight = 0.0
    if resume and os.path.exists(speed_weight_save_path):
        with open(speed_weight_save_path) as f:
            init_speed_weight = float(f.read().strip())
        print(f"  speed_weight 복원: {init_speed_weight:.2f}")

    learning_starts = 30_000

    if resume and os.path.exists(resume + ".zip"):
        print(f"\n이어서 학습: {resume}")
        model = SAC.load(resume, env=vec_env, device="cpu", tensorboard_log=log_dir)
        model.learning_starts = model.num_timesteps + learning_starts
        print(f"  learning_starts 재설정: {model.learning_starts:,}")
        from stable_baselines3.common.utils import get_schedule_fn
        finetune_lr = 3e-4
        model.learning_rate = finetune_lr
        model.lr_schedule   = get_schedule_fn(finetune_lr)
        for opt in [model.policy.actor.optimizer,
                    model.policy.critic.optimizer,
                    model.ent_coef_optimizer]:
            for pg in opt.param_groups:
                pg["lr"] = finetune_lr
        print(f"  LR fine-tune: {finetune_lr}")
    else:
        policy_kwargs = dict(
            features_extractor_class=ObstacleAwareExtractor,
            features_extractor_kwargs={},
            net_arch=[256, 256, 256],
        )
        model = SAC(
            policy="MultiInputPolicy",
            env=vec_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.95,
            ent_coef="auto_0.1",
            target_entropy=-1.0,
            learning_starts=learning_starts,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device="cpu",
            verbose=1,
        )

    # 모든 환경에 초기 curriculum 설정
    for vec in [vec_env, eval_env]:
        envs = vec.envs if hasattr(vec, "envs") else vec.venv.envs
        for env in envs:
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            unwrapped.set_success_threshold(init_threshold, init_ori_threshold)
            unwrapped.set_init_range(init_range)
            unwrapped.set_max_obs_count(init_max_obs_count)
            unwrapped.set_speed_weight(init_speed_weight)

    print(f"\n환경 정보:")
    print(f"  Env ID:             M1013Reach-v0")
    print(f"  Obs space:          27D robot + 70D obstacles")
    print(f"  병렬 환경 수:       {n_envs}")
    print(f"  총 학습 스텝:       {total_timesteps:,}")
    print(f"  모델 파라미터 수:   {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )

    warmup_eps = 2000 if resume else 0
    curriculum_callback = CurriculumCallback(
        print_freq=500,
        init_threshold=init_threshold,
        init_ori_threshold=init_ori_threshold,
        init_range=init_range,
        init_max_obs_count=init_max_obs_count,
        init_speed_weight=init_speed_weight,
        threshold_save_path=threshold_save_path,
        ori_threshold_save_path=ori_threshold_save_path,
        init_range_save_path=init_range_save_path,
        max_obs_count_save_path=max_obs_count_save_path,
        speed_weight_save_path=speed_weight_save_path,
        eval_vec_env=eval_env,
        warmup_episodes=warmup_eps,
        verbose=1,
    )
    vecnorm_callback   = VecNormSaveCallback(vecnorm_path, save_freq=10_000)
    ent_floor_callback = EntCoefFloorCallback(min_ent_coef=0.03)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, curriculum_callback, vecnorm_callback, ent_floor_callback],
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, "sac_m1013_final"))
    vec_env.save(vecnorm_path)
    print(f"\n학습 완료! → {model_dir}/sac_m1013_final.zip")

    vec_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M1013 SAC + HER 학습 (Torque)")
    parser.add_argument("--steps",  type=int, default=2_000_000)
    parser.add_argument("--envs",   type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 모델 경로 (예: models/torque/best_model)")
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.envs, resume=args.resume)
