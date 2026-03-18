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

LOG_DIR = "./logs"
MODEL_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class CurriculumCallback(BaseCallback):
    """
    성공률 기반 Curriculum Learning 콜백.
    - 성공률 >= 50% : curriculum level +0.05
    - 성공률 <  20% : curriculum level -0.02
    """

    def __init__(self, print_freq=500, init_level=0.0, verbose=0):
        super().__init__(verbose)
        self.successes = []
        self.episodes = 0
        self._last_logged_ep = 0
        self.print_freq = print_freq
        self.curriculum_level = init_level

    def _set_level(self, level: float):
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))
        for env in self.training_env.envs:
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            unwrapped.set_curriculum_level(self.curriculum_level)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # "episode" key is added by Monitor only when an episode completes
            if "episode" in info:
                self.successes.append(float(info.get("success", False)))
                self.episodes += 1

        if self.episodes >= self._last_logged_ep + self.print_freq:
            recent = self.successes[-self.print_freq:]
            rate = np.mean(recent) * 100

            if rate >= 50.0:
                self._set_level(self.curriculum_level + 0.05)
            elif rate < 20.0 and self.curriculum_level > 0.0:
                self._set_level(self.curriculum_level - 0.02)

            max_r = 0.08 + (0.80 - 0.08) * self.curriculum_level
            print(
                f"  [Ep {self.episodes:>7d}] "
                f"성공률: {rate:5.1f}% | "
                f"level: {self.curriculum_level:.2f} (max_r={max_r:.2f}m)"
            )
            self._last_logged_ep = self.episodes
            self.logger.record("train/success_rate", rate)
            self.logger.record("train/curriculum_level", self.curriculum_level)

        return True


def make_env(env_id="M1013Reach-v0"):
    env = gym.make(env_id)
    return Monitor(env)


def train(total_timesteps: int = 1_000_000, n_envs: int = 8, resume: str = None):
    print("=" * 65)
    print("  Doosan M1013 Reach Task — PPO 학습")
    print("=" * 65)

    vec_env  = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    if resume and os.path.exists(resume + ".zip"):
        print(f"이어서 학습: {resume}")
        model = PPO.load(resume, env=vec_env, device="cpu", tensorboard_log=LOG_DIR)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lambda f: 3e-4 * f,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                log_std_init=-1.0,
            ),
            tensorboard_log=LOG_DIR,
            device="cpu",
            verbose=1,
        )

    print(f"\n환경 정보:")
    print(f"  Env ID:             M1013Reach-v0")
    print(f"  Observation space:  {vec_env.observation_space}")
    print(f"  Action space:       {vec_env.action_space}")
    print(f"  병렬 환경 수:       {n_envs}")
    print(f"  총 학습 스텝:       {total_timesteps:,}")
    print(f"  모델 파라미터 수:   {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=MODEL_DIR,
        name_prefix="ppo_m1013",
    )
    success_callback = CurriculumCallback(print_freq=500, init_level=0.0, verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, success_callback],
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_m1013_final"))
    print(f"\n학습 완료! → {MODEL_DIR}/ppo_m1013_final.zip")

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
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.envs, resume=args.resume)
