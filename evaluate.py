"""
M1013 학습된 모델 평가 스크립트
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import robot_env  # M1013Reach-v0 등록


def evaluate(model_path: str, n_episodes: int = 20, render: bool = False):
    print(f"\n모델 로드: {model_path}")
    model = PPO.load(model_path)

    env = gym.make("M1013Reach-v0", render_mode="human" if render else None)

    successes, distances, ep_rewards = [], [], []

    print(f"\n{'─'*55}")
    print(f"  평가: {n_episodes} 에피소드 | 성공 기준: 5cm 이내")
    print(f"{'─'*55}")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        min_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            min_dist = min(min_dist, info["distance"])

        successes.append(float(info.get("success", False)))
        distances.append(min_dist)
        ep_rewards.append(ep_reward)

        status = "✓" if info.get("success") else "✗"
        print(f"  [{status}] Ep {ep+1:2d}: 최소거리={min_dist:.3f}m | 보상={ep_reward:7.1f}")

    print(f"\n{'─'*55}")
    print(f"  성공률:        {np.mean(successes)*100:.1f}%  ({int(sum(successes))}/{n_episodes})")
    print(f"  평균 최소거리: {np.mean(distances)*100:.1f} cm")
    print(f"  평균 보상:     {np.mean(ep_rewards):.2f}")
    print(f"{'─'*55}")

    env.close()


def demo_random(n_steps: int = 300):
    """랜덤 에이전트로 환경 동작 확인."""
    print("\nM1013 환경 확인 (랜덤 에이전트)...")
    env = gym.make("M1013Reach-v0")
    obs, info = env.reset()

    print(f"  Obs dim:    {obs.shape[0]}")
    print(f"  Action dim: {env.action_space.shape[0]}")
    print(f"  Target:     {info['target_pos']}")
    print(f"  EE start:   {info['ee_pos']}")

    total_reward = 0.0
    ep = 0
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            ep += 1
            print(f"  Ep {ep}: 스텝={i+1} 누적보상={total_reward:.1f} 최소거리={info['distance']:.3f}m")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
    print("랜덤 데모 완료!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    if args.random:
        demo_random()
    elif os.path.exists(args.model + ".zip") or os.path.exists(args.model):
        evaluate(args.model, n_episodes=args.episodes, render=args.render)
    else:
        print(f"모델 없음: {args.model}")
        print("먼저 python train.py 를 실행하세요.\n")
        demo_random()
