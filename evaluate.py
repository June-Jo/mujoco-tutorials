"""
M1013 학습된 모델 평가 스크립트
"""

import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import robot_env  # M1013Reach-v0 등록
from robot_env import RobotArmBaseEnv


def evaluate(model_path: str, n_episodes: int = 20, render: bool = True,
             control_mode: str = "position", speed: float = 1.0):
    print(f"\n모델 로드: {model_path}")
    model = PPO.load(model_path)

    env_id = "M1013Reach-v0" if control_mode == "position" else "M1013Reach-Torque-v0"
    env = gym.make(env_id, render_mode="human" if render else None)

    # success_threshold 복원
    model_dir = os.path.join("models", control_mode)
    threshold_path = os.path.join(model_dir, "success_threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = float(f.read().strip())
        env.unwrapped.set_success_threshold(threshold)
        print(f"  Success threshold: {threshold*100:.2f}cm")
    else:
        threshold = RobotArmBaseEnv.SUCCESS_THRESHOLD_INIT
        print(f"  Success threshold: {threshold*100:.2f}cm (기본값)")

    successes, distances, ep_rewards = [], [], []

    print(f"\n{'─'*55}")
    print(f"  평가: {n_episodes} 에피소드 | 성공 기준: {threshold*100:.1f}cm 이내")
    print(f"{'─'*55}")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        min_dist = float("inf")

        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            min_dist = min(min_dist, info["distance"])
            step += 1
            if render:
                print(f"\r  step={step:4d} | dist={info['distance']:.3f}m", end="", flush=True)
                time.sleep(0.02 / speed)  # 실시간(1.0) 기준 50Hz 유지
        if render:
            print()

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


def demo_random(n_steps: int = 300, control_mode: str = "position"):
    """랜덤 에이전트로 환경 동작 확인."""
    env_id = "M1013Reach-v0" if control_mode == "position" else "M1013Reach-Torque-v0"
    print(f"\nM1013 환경 확인 (랜덤 에이전트, {control_mode})...")
    env = gym.make(env_id)
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
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_false", help="화면 렌더링 여부 (기본: True)")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--torque", action="store_true", help="토크 제어 모드")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="재생 속도 배율 (기본 1.0=실시간, 2.0=2배속)")
    args = parser.parse_args()

    control_mode = "torque" if args.torque else "position"
    default_model = f"models/{control_mode}/best_model"
    model_path = args.model or default_model

    if args.random:
        demo_random(control_mode=control_mode)
    elif os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        evaluate(model_path, n_episodes=args.episodes, render=args.render,
                 control_mode=control_mode, speed=args.speed)
    else:
        print(f"모델 없음: {model_path}")
        print("먼저 python train.py 를 실행하세요.\n")
        demo_random(control_mode=control_mode)
