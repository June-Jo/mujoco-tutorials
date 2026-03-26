"""
Doosan M1013 - SAC + HER 학습 스크립트 (Torque 제어 전용)

설계:
  - Observation: qpos(6) + qvel(6) + ee_pose(7) + ee_vel(7) = 26D
  - Goal: ee_pos(3) + ee_quat(4) = 7D
  - Obstacles: 10 × 7D = 70D (별도 MLP 인코더 처리)
  - Reward: -(pos_dist + 1.0 × angle) + 성공 보너스(+10) + 충돌 패널티(-5)
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
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import robot_env  # M1013Reach-v0 등록
from robot_env import (M1013Env as RobotArmEnv, MAX_OBSTACLES,
                       OBS_UNLOCK_THRESHOLD)

LOG_DIR_BASE   = "./logs"
MODEL_DIR_BASE = "./models/torque"


# ── Custom Feature Extractor ──────────────────────────────────────────────

class ObstacleAwareExtractor(BaseFeaturesExtractor):
    """
    장애물 정보를 별도 MLP로 처리한 뒤 로봇 상태와 concat.

    구조:
      robot_obs(26) + achieved_goal(7) + desired_goal(7) = 40D → robot_net → 256D
      obstacles(70D) → obstacle_net → 64D
      concat → 320D (features_dim)
    """

    def __init__(self, observation_space: spaces.Dict,
                 robot_out: int = 512, obstacle_out: int = 128):
        robot_dim = (
            observation_space["observation"].shape[0]    # 27
            + observation_space["achieved_goal"].shape[0]  # 7
            + observation_space["desired_goal"].shape[0]   # 7
        )  # = 41
        obstacle_dim = observation_space["obstacles"].shape[0]  # 70
        features_dim = robot_out + obstacle_out  # 640

        super().__init__(observation_space, features_dim)

        self.robot_net = nn.Sequential(
            nn.Linear(robot_dim, 512),
            nn.ReLU(),
            nn.Linear(512, robot_out),
            nn.ReLU(),
        )
        self.obstacle_net = nn.Sequential(
            nn.Linear(obstacle_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obstacle_out),
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
    pos/ori threshold, init_range, max_obs_count 자동 조정.

    성공률 >= 85%:
      - pos / ori / obs / init_range 중 하나를 랜덤 선택해 난이도 상승
        · pos: success_threshold × 0.8
        · ori: ori_threshold × 0.8
        · obs: max_obs_count + 1 (최대치면 선택지에서 제외)
        · init_range: init_range × 1.5 (최대치면 선택지에서 제외)

    성공률 < 20%:
      - pos/ori threshold × 1.2  (나머지 유지)
    """

    def __init__(self, print_freq=500,
                 init_threshold=0.30, init_ori_threshold=None,
                 init_range=None, init_max_obs_count=0,
                 threshold_save_path=None, ori_threshold_save_path=None,
                 init_range_save_path=None, max_obs_count_save_path=None,
                 eval_vec_env=None,
                 warmup_episodes=0,
                 ent_floor_callback=None,
                 stagnation_windows=10,
                 verbose=0):
        super().__init__(verbose)
        self.successes       = []
        self.episodes        = 0
        self._last_logged_ep = 0
        self.print_freq      = print_freq
        self.warmup_episodes = warmup_episodes
        self.ent_floor_callback       = ent_floor_callback
        self.stagnation_windows       = stagnation_windows
        self._best_rate               = 0.0
        self._windows_no_improvement  = 0

        self.success_threshold = init_threshold
        self.ori_threshold     = (init_ori_threshold
                                  if init_ori_threshold is not None
                                  else RobotArmEnv.ORI_THRESHOLD_INIT)
        self.init_range        = (init_range
                                  if init_range is not None
                                  else RobotArmEnv.INIT_RANGE_INIT)
        self.max_obs_count     = int(init_max_obs_count)

        self.threshold_save_path      = threshold_save_path
        self.ori_threshold_save_path  = ori_threshold_save_path
        self.init_range_save_path     = init_range_save_path
        self.max_obs_count_save_path  = max_obs_count_save_path
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
                        init_range: float = None, max_obs_count: int = None):
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

        all_envs = self._unwrap_envs(self.training_env)
        if self.eval_vec_env is not None:
            all_envs += self._unwrap_envs(self.eval_vec_env)

        for env in all_envs:
            env.set_success_threshold(self.success_threshold, self.ori_threshold)
            env.set_init_range(self.init_range)
            env.set_max_obs_count(self.max_obs_count)

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

        # 정체 감지: 카운터만 유지 (부스트 제거 — ent_coef 상승이 오히려 역효과)
        if rate > self._best_rate + 2.0:
            self._best_rate = rate
            self._windows_no_improvement = 0
        else:
            self._windows_no_improvement += 1

        if rate >= 85.0:
            # pos / ori / obs / init_range 중 하나를 랜덤 선택해 난이도 상승
            options = ["pos", "ori"]
            if self.max_obs_count < MAX_OBSTACLES:
                options.append("obs")
            if self.init_range < RobotArmEnv.INIT_RANGE_MAX:
                options.append("init_range")
            choice = np.random.choice(options)

            if choice == "obs":
                self._set_curriculum(
                    self.success_threshold,
                    self.ori_threshold,
                    max_obs_count=self.max_obs_count + 1,
                )
                print(f"  [Curriculum] 전진: obs {self.max_obs_count}/{MAX_OBSTACLES}")
            elif choice == "pos":
                self._set_curriculum(
                    self.success_threshold * 0.8,
                    self.ori_threshold,
                )
                print(f"  [Curriculum] 전진: pos → {self.success_threshold*100:.2f}cm")
            elif choice == "ori":
                self._set_curriculum(
                    self.success_threshold,
                    self.ori_threshold * 0.8,
                )
                print(f"  [Curriculum] 전진: ori → {np.degrees(self.ori_threshold):.2f}°")
            else:  # init_range
                self._set_curriculum(
                    self.success_threshold,
                    self.ori_threshold,
                    init_range=self.init_range * 1.5,
                )
                print(f"  [Curriculum] 전진: init_range → ±{np.degrees(self.init_range):.0f}°")
            # 새 스테이지 진입 시 정체 카운터 리셋 (부스트 없음)
            if self.ent_floor_callback is not None:
                self._best_rate = 0.0
                self._windows_no_improvement = 0
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
            f"ent: {float(self.model.target_entropy):.2f}"
        )
        self._last_logged_ep = self.episodes
        self.logger.record("train/success_rate",        rate)
        self.logger.record("train/success_threshold",   self.success_threshold)
        self.logger.record("train/ori_threshold_deg",   np.degrees(self.ori_threshold))
        self.logger.record("train/init_range_deg",      np.degrees(self.init_range))
        self.logger.record("train/max_obs_count",       self.max_obs_count)
        self.logger.record("train/target_entropy",      float(self.model.target_entropy))
        return True


class EntCoefFloorCallback(BaseCallback):
    """적응형 ent_coef floor 콜백.

    평상시: base_floor (낮음) — SAC가 자유롭게 exploit 가능
    부스트 중: boost_floor (높음) — 커리큘럼 전진 or 정체 시 일시적 탐색 강화

    boost()를 호출하면 boost_episodes 동안 floor를 높임.
    """

    def __init__(self, base_floor: float = 0.05, boost_floor: float = 0.15,
                 boost_episodes: int = 2000, verbose=0):
        super().__init__(verbose)
        self.base_floor     = base_floor
        self.boost_floor    = boost_floor
        self.boost_episodes = boost_episodes
        self._boost_remaining = 0
        self._active_floor    = base_floor

    def boost(self, reason: str = ""):
        """일시적으로 floor를 boost_floor로 올림."""
        self._boost_remaining = self.boost_episodes
        self._active_floor    = self.boost_floor
        print(
            f"  [EntCoef] floor 부스트 → {self.boost_floor} "
            f"({self.boost_episodes}ep)"
            + (f" — {reason}" if reason else "")
        )

    def is_boosting(self) -> bool:
        return self._boost_remaining > 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and self._boost_remaining > 0:
                self._boost_remaining -= 1
                if self._boost_remaining == 0:
                    self._active_floor = self.base_floor
                    print(f"  [EntCoef] 부스트 종료 → floor {self.base_floor}")
        self.model.log_ent_coef.data.clamp_(min=math.log(self._active_floor))
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

    learning_starts = 30_000

    model = None
    if resume and os.path.exists(resume + ".zip"):
        print(f"\n이어서 학습: {resume}")
        try:
            model = SAC.load(resume, env=vec_env, device="cpu", tensorboard_log=log_dir)
            # model.learn()이 reset_num_timesteps=True(기본값)로 num_timesteps를 0으로 리셋하므로
            # model.num_timesteps 기반으로 설정하면 안 됨 → 단순히 learning_starts만 사용
            model.learning_starts = learning_starts
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
            # B: target_entropy 업데이트
            model.target_entropy = -1.0
            print(f"  target_entropy 재설정: {model.target_entropy}")
            # C: 액션 노이즈 추가
            model.action_noise = NormalActionNoise(mean=np.zeros(6), sigma=0.1 * np.ones(6))
            print(f"  액션 노이즈 추가: N(0, 0.1)")
        except Exception as e:
            print(f"  ⚠️  모델 로드 실패 (아키텍처 불일치 등): {e}")
            print(f"  새 모델로 시작합니다.")
            model = None

    if model is None:
        policy_kwargs = dict(
            features_extractor_class=ObstacleAwareExtractor,
            features_extractor_kwargs={},
            net_arch=[512, 512, 512],  # 256 → 512 (네트워크 확장)
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
            batch_size=512,
            tau=0.005,
            gamma=0.95,
            ent_coef="auto_0.2",           # 초기값 상향 (floor 0.15보다 여유)
            target_entropy=-1.0,           # B: -3.0 → -1.0 (높은 엔트로피 목표)
            action_noise=NormalActionNoise( # C: 롤아웃 액션 노이즈
                mean=np.zeros(6),
                sigma=0.1 * np.ones(6),
            ),
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

    print(f"\n환경 정보:")
    print(f"  Env ID:             M1013Reach-v0")
    print(f"  Obs space:          26D robot + 70D obstacles")
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
    ent_floor_callback = EntCoefFloorCallback(base_floor=0.05, boost_floor=0.15,
                                              boost_episodes=2000)
    curriculum_callback = CurriculumCallback(
        print_freq=500,
        init_threshold=init_threshold,
        init_ori_threshold=init_ori_threshold,
        init_range=init_range,
        init_max_obs_count=init_max_obs_count,
        threshold_save_path=threshold_save_path,
        ori_threshold_save_path=ori_threshold_save_path,
        init_range_save_path=init_range_save_path,
        max_obs_count_save_path=max_obs_count_save_path,
        eval_vec_env=eval_env,
        warmup_episodes=warmup_eps,
        ent_floor_callback=ent_floor_callback,
        stagnation_windows=10,
        verbose=1,
    )
    vecnorm_callback   = VecNormSaveCallback(vecnorm_path, save_freq=10_000)

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
