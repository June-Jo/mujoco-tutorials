"""
Doosan M1013 - SAC 학습 스크립트 (Torque 제어 전용)

설계:
  - Observation: qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7) + target_pos(3) + target_quat(4) = 33D
  - Reward: -(dist + angle) + progress + proximity×exp(-vel×5) + 성공(+10) + 자가충돌(-5)
  - VecNormalize: obs 정규화 ON, reward 정규화 OFF
  - MlpPolicy: [256, 256] MLP
"""

import os
import math
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym

import robot_env  # M1013Reach-v0 등록
from robot_env import M1013Env as RobotArmEnv

LOG_DIR_BASE   = "./logs"
MODEL_DIR_BASE = "./models/torque"


# ── Callbacks ────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    성공률 기반 Curriculum 콜백.
    pos/ori threshold, init_range 자동 조정.

    성공률 >= 85%:
      - pos / ori / init_range 중 하나를 랜덤 선택해 난이도 상승
        · pos: success_threshold × 0.8
        · ori: ori_threshold × 0.8
        · init_range: init_range × 1.5 (최대치면 선택지에서 제외)

    성공률 < 20%:
      - pos/ori threshold × 1.2  (나머지 유지)
    """

    def __init__(self, print_freq=500,
                 init_threshold=0.30, init_ori_threshold=None,
                 init_range=None,
                 threshold_save_path=None, ori_threshold_save_path=None,
                 init_range_save_path=None,
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
        self.ent_floor_callback      = ent_floor_callback
        self.stagnation_windows      = stagnation_windows
        self._best_rate              = 0.0
        self._windows_no_improvement = 0

        self.success_threshold = init_threshold
        self.ori_threshold     = (init_ori_threshold
                                  if init_ori_threshold is not None
                                  else RobotArmEnv.ORI_THRESHOLD_INIT)
        self.init_range        = (init_range
                                  if init_range is not None
                                  else RobotArmEnv.INIT_RANGE_INIT)

        self.threshold_save_path     = threshold_save_path
        self.ori_threshold_save_path = ori_threshold_save_path
        self.init_range_save_path    = init_range_save_path
        self.eval_vec_env            = eval_vec_env

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
                        init_range: float = None):
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

        all_envs = self._unwrap_envs(self.training_env)
        if self.eval_vec_env is not None:
            all_envs += self._unwrap_envs(self.eval_vec_env)

        for env in all_envs:
            env.set_success_threshold(self.success_threshold, self.ori_threshold)
            env.set_init_range(self.init_range)

        if self.threshold_save_path:
            with open(self.threshold_save_path, "w") as f:
                f.write(str(self.success_threshold))
        if self.ori_threshold_save_path:
            with open(self.ori_threshold_save_path, "w") as f:
                f.write(str(self.ori_threshold))
        if self.init_range_save_path:
            with open(self.init_range_save_path, "w") as f:
                f.write(str(self.init_range))

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

        # 정체 감지
        if rate > self._best_rate + 2.0:
            self._best_rate = rate
            self._windows_no_improvement = 0
        else:
            self._windows_no_improvement += 1

        # 정체 시 탐색 부스트 (성공률 개선 없이 N 윈도우 경과)
        if (self._windows_no_improvement >= self.stagnation_windows
                and self.ent_floor_callback is not None
                and not self.ent_floor_callback.is_boosting()):
            self.ent_floor_callback.boost(reason="정체 감지")
            self._windows_no_improvement = 0

        if rate >= 85.0:
            options = ["pos", "ori"]
            if self.init_range < RobotArmEnv.INIT_RANGE_MAX:
                options.append("init_range")
            choice = np.random.choice(options)

            if choice == "pos":
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
            # 새 스테이지 진입 시 정체 카운터 리셋
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
            f"ent: {float(self.model.target_entropy):.2f}"
        )
        self._last_logged_ep = self.episodes
        self.logger.record("train/success_rate",      rate)
        self.logger.record("train/success_threshold", self.success_threshold)
        self.logger.record("train/ori_threshold_deg", np.degrees(self.ori_threshold))
        self.logger.record("train/init_range_deg",    np.degrees(self.init_range))
        self.logger.record("train/target_entropy",    float(self.model.target_entropy))
        return True


class EntCoefFloorCallback(BaseCallback):
    """적응형 ent_coef floor 콜백.

    평상시: base_floor (낮음) — SAC가 자유롭게 exploit 가능
    부스트 중: boost_floor (높음) — 커리큘럼 전진 or 정체 시 일시적 탐색 강화

    boost()를 호출하면 boost_episodes 동안 floor를 높임.
    """

    def __init__(self, base_floor: float = 0.0, boost_floor: float = 0.15,
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
                    label = f"{self.base_floor}" if self.base_floor > 0 else "없음"
                    print(f"  [EntCoef] 부스트 종료 → floor {label}")
        if self._active_floor > 0:
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
    print("  Doosan M1013 — SAC (Torque, Reach Only)")
    print("=" * 65)

    vec_env_raw  = make_vec_env(make_env, n_envs=n_envs)
    eval_env_raw = make_vec_env(make_env, n_envs=1)

    vecnorm_path            = os.path.join(model_dir, "vecnormalize.pkl")
    threshold_save_path     = os.path.join(model_dir, "success_threshold.txt")
    ori_threshold_save_path = os.path.join(model_dir, "ori_threshold.txt")
    init_range_save_path    = os.path.join(model_dir, "init_range.txt")

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

    learning_starts = 30_000

    model = None
    if resume and os.path.exists(resume + ".zip"):
        print(f"\n이어서 학습: {resume}")
        try:
            model = SAC.load(resume, env=vec_env, device="cpu", tensorboard_log=log_dir)
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
            model.target_entropy = -3.0
            print(f"  target_entropy 재설정: {model.target_entropy}")
            model.action_noise = NormalActionNoise(mean=np.zeros(6), sigma=0.1 * np.ones(6))
            print(f"  액션 노이즈 추가: N(0, 0.1)")
        except Exception as e:
            print(f"  ⚠️  모델 로드 실패 (아키텍처 불일치 등): {e}")
            print(f"  새 모델로 시작합니다.")
            model = None

    if model is None:
        policy_kwargs = dict(
            net_arch=[256, 256],
        )
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=512,
            tau=0.005,
            gamma=0.95,
            ent_coef="auto_0.2",
            target_entropy=-3.0,
            action_noise=NormalActionNoise(
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

    print(f"\n환경 정보:")
    print(f"  Env ID:             M1013Reach-v0")
    print(f"  Obs space:          33D flat (robot 26D + target 7D)")
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
    ent_floor_callback = EntCoefFloorCallback(base_floor=0.0, boost_floor=0.15,
                                              boost_episodes=2000)
    curriculum_callback = CurriculumCallback(
        print_freq=500,
        init_threshold=init_threshold,
        init_ori_threshold=init_ori_threshold,
        init_range=init_range,
        threshold_save_path=threshold_save_path,
        ori_threshold_save_path=ori_threshold_save_path,
        init_range_save_path=init_range_save_path,
        eval_vec_env=eval_env,
        warmup_episodes=warmup_eps,
        ent_floor_callback=ent_floor_callback,
        stagnation_windows=10,
        verbose=1,
    )
    vecnorm_callback = VecNormSaveCallback(vecnorm_path, save_freq=10_000)

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

    parser = argparse.ArgumentParser(description="M1013 SAC 학습 (Torque, Reach Only)")
    parser.add_argument("--steps",  type=int, default=2_000_000)
    parser.add_argument("--envs",   type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 모델 경로 (예: models/torque/best_model)")
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.envs, resume=args.resume)
