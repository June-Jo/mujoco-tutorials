"""
Doosan M1013 Robot Arm - PPO 학습 스크립트
stable-baselines3 + MuJoCo
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import robot_env  # M1013Reach-v0 등록
from robot_env import RobotArmBaseEnv as RobotArmEnv

LOG_DIR_BASE   = "./logs"
MODEL_DIR_BASE = "./models"


class CurriculumCallback(BaseCallback):
    """
    성공률 기반 Curriculum Learning 콜백.
    - 성공률 >= 70% : success_threshold 절반 (더 어렵게)
    - 성공률 <  20% : success_threshold 2배 (더 쉽게, 최대 30cm)
    - 최소 threshold: 0.1mm
    """

    def __init__(self, print_freq=500, init_threshold=0.30, threshold_save_path=None, verbose=0):
        super().__init__(verbose)
        self.successes = []
        self.episodes = 0
        self._last_logged_ep = 0
        self.print_freq = print_freq
        self.success_threshold = init_threshold
        self.threshold_save_path = threshold_save_path

    def _set_threshold(self, threshold: float):
        self.success_threshold = float(np.clip(
            threshold,
            RobotArmEnv.SUCCESS_THRESHOLD_MIN,
            RobotArmEnv.SUCCESS_THRESHOLD_MAX,
        ))
        for env in self.training_env.envs:
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            unwrapped.set_success_threshold(self.success_threshold)
        if self.threshold_save_path:
            with open(self.threshold_save_path, "w") as f:
                f.write(str(self.success_threshold))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # "episode" key is added by Monitor only when an episode completes
            if "episode" in info:
                self.successes.append(float(info.get("success", False)))
                self.episodes += 1

        if self.episodes >= self._last_logged_ep + self.print_freq:
            recent = self.successes[-self.print_freq:]
            rate = np.mean(recent) * 100

            if rate >= 70.0:
                self._set_threshold(self.success_threshold * 0.5)
            elif rate < 20.0:
                self._set_threshold(self.success_threshold * 2.0)

            print(
                f"  [Ep {self.episodes:>7d}] "
                f"성공률: {rate:5.1f}% | "
                f"threshold: {self.success_threshold*100:.2f}cm"
            )
            self._last_logged_ep = self.episodes
            self.logger.record("train/success_rate", rate)
            self.logger.record("train/success_threshold", self.success_threshold)

        return True


def make_env(env_id):
    env = gym.make(env_id)
    return Monitor(env)


def train(total_timesteps: int = 1_000_000, n_envs: int = 8, resume: str = None,
          control_mode: str = "position"):
    env_id    = "M1013Reach-v0" if control_mode == "position" else "M1013Reach-Torque-v0"
    log_dir   = os.path.join(LOG_DIR_BASE,   control_mode)
    model_dir = os.path.join(MODEL_DIR_BASE, control_mode)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 65)
    print(f"  Doosan M1013 Reach Task — PPO 학습 ({control_mode})")
    print("=" * 65)

    vec_env  = make_vec_env(lambda: make_env(env_id), n_envs=n_envs)
    eval_env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    threshold_save_path = os.path.join(model_dir, "success_threshold.txt")

    # resume 시 저장된 success_threshold 복원
    init_threshold = RobotArmEnv.SUCCESS_THRESHOLD_INIT
    if resume and os.path.exists(threshold_save_path):
        with open(threshold_save_path) as f:
            init_threshold = float(f.read().strip())
        print(f"  Success threshold 복원: {init_threshold*100:.2f}cm")

    if resume and os.path.exists(resume + ".zip"):
        print(f"이어서 학습: {resume}")
        model = PPO.load(resume, env=vec_env, device="cpu", tensorboard_log=log_dir)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                log_std_init=0.0,
            ),
            tensorboard_log=log_dir,
            device="cpu",
            verbose=1,
        )

    print(f"\n환경 정보:")
    print(f"  Env ID:             {env_id}")
    print(f"  Observation space:  {vec_env.observation_space}")
    print(f"  Action space:       {vec_env.action_space}")
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
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=model_dir,
        name_prefix="ppo_m1013",
    )
    # 모든 환경에 초기 threshold 설정
    for env in vec_env.envs:
        unwrapped = env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        unwrapped.set_success_threshold(init_threshold)

    success_callback = CurriculumCallback(print_freq=500, init_threshold=init_threshold,
                                          threshold_save_path=threshold_save_path, verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, success_callback],
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, "ppo_m1013_final"))
    print(f"\n학습 완료! → {model_dir}/ppo_m1013_final.zip")

    vec_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M1013 PPO 학습")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="총 학습 스텝 (기본값: 1,000,000)")
    parser.add_argument("--envs",  type=int, default=8,
                        help="병렬 환경 수 (기본값: 8)")
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 모델 경로 (예: models/best_model)")
    parser.add_argument("--torque", action="store_true",
                        help="토크 제어 모드로 학습 (기본값: position 제어)")
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.envs, resume=args.resume,
          control_mode="torque" if args.torque else "position")
