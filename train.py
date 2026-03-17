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


class SuccessRateCallback(BaseCallback):
    """에피소드 성공률을 주기적으로 출력."""

    def __init__(self, print_freq=200, verbose=0):
        super().__init__(verbose)
        self.successes = []
        self.episodes = 0
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.successes.append(float(info["success"]))
                self.episodes += 1

        if self.episodes > 0 and self.episodes % self.print_freq == 0:
            recent = self.successes[-400:] if len(self.successes) >= 400 else self.successes
            rate = np.mean(recent) * 100
            print(f"  [Ep {self.episodes:>7d}] 성공률(최근 {len(recent)}): {rate:5.1f}%")
            self.logger.record("train/success_rate", rate)

        return True


def make_env(env_id="M1013Reach-v0"):
    env = gym.make(env_id)
    return Monitor(env)


def train(total_timesteps: int = 1_000_000, n_envs: int = 8):
    print("=" * 65)
    print("  Doosan M1013 Reach Task — PPO 학습")
    print("=" * 65)

    vec_env  = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        # ── 학습률 스케줄 (초반 빠르게, 후반 안정적으로)
        learning_rate=lambda f: 3e-4 * f,   # f: 1→0 as training progresses
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
        device="cpu",   # MLP policy는 CPU가 더 빠름
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
    success_callback = SuccessRateCallback(print_freq=500, verbose=1)

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
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.envs)
