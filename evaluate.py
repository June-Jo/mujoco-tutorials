"""
M1013 학습된 모델 평가 스크립트 (SAC + HER, Torque 제어)
"""

import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import robot_env  # M1013Reach-v0 등록
from robot_env import M1013Env


def _unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def evaluate(model_path: str, n_episodes: int = 20,
             render: bool = False, speed: float = 1.0,
             max_obs_override: int = None):
    model_dir    = "./models/torque"
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

    render_mode = "human" if render else None
    vec_env = DummyVecEnv([lambda: Monitor(gym.make("M1013Reach-v0", render_mode=render_mode))])

    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"  VecNormalize 로드: {vecnorm_path}")
    else:
        print("  VecNormalize 없음 — raw obs 사용")

    print(f"\n모델 로드: {model_path}")
    model = SAC.load(model_path, env=vec_env)

    # Curriculum 상태 복원
    def _read_float(path, default):
        if os.path.exists(path):
            with open(path) as f:
                return float(f.read().strip())
        return default

    def _read_int(path, default):
        if os.path.exists(path):
            with open(path) as f:
                return int(f.read().strip())
        return default

    threshold     = _read_float(os.path.join(model_dir, "success_threshold.txt"),
                                M1013Env.SUCCESS_THRESHOLD_INIT)
    ori_threshold = _read_float(os.path.join(model_dir, "ori_threshold.txt"),
                                M1013Env.ORI_THRESHOLD_INIT)
    init_range    = _read_float(os.path.join(model_dir, "init_range.txt"),
                                M1013Env.INIT_RANGE_INIT)
    max_obs_count = _read_int(os.path.join(model_dir, "max_obs_count.txt"), 0)

    inner_env = vec_env.venv.envs[0] if isinstance(vec_env, VecNormalize) else vec_env.envs[0]
    unwrapped = _unwrap(inner_env)
    unwrapped.set_success_threshold(threshold, ori_threshold)
    unwrapped.set_init_range(init_range)
    if max_obs_override is not None:
        unwrapped.set_max_obs_count(max_obs_override)
        unwrapped.fixed_obs_count = max_obs_override  # 정확히 N개 고정
    else:
        unwrapped.set_max_obs_count(max_obs_count)

    print(f"  pos threshold:  {threshold*100:.2f} cm")
    print(f"  ori threshold:  {np.degrees(ori_threshold):.1f}°")
    print(f"  init range:     ±{np.degrees(init_range):.0f}°")
    print(f"  max obstacles:  {max_obs_count}")

    successes, distances, ep_rewards, angles, collisions = [], [], [], [], []

    print(f"\n{'─'*65}")
    print(f"  평가: {n_episodes} 에피소드 | "
          f"pos≤{threshold*100:.1f}cm / ori≤{np.degrees(ori_threshold):.1f}° | "
          f"장애물 최대 {max_obs_count}개")
    print(f"{'─'*65}")

    for ep in range(n_episodes):
        obs      = vec_env.reset()
        done     = False
        ep_reward    = 0.0
        min_dist     = float("inf")
        min_angle    = float("inf")
        step         = 0
        info         = {}
        ep_collision = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)
            done      = bool(done_arr[0])
            ep_reward += float(reward[0])
            info       = info_arr[0]
            dist       = info.get("distance",    float("inf"))
            angle      = info.get("angle_error", float("inf"))
            min_dist   = min(min_dist, dist)
            min_angle  = min(min_angle, angle)
            step      += 1
            if info.get("collision"):
                ep_collision = True
            if render:
                coll_flag = " 💥" if info.get("collision") else ""
                print(f"\r  step={step:4d} | dist={dist:.3f}m | "
                      f"angle={np.degrees(angle):.1f}°{coll_flag}",
                      end="", flush=True)
                time.sleep(0.02 / speed)

        if render:
            print()

        successes.append(float(info.get("success", False)))
        distances.append(min_dist)
        ep_rewards.append(ep_reward)
        angles.append(min_angle)
        collisions.append(float(ep_collision))

        status = "✓" if info.get("success") else "✗"
        coll   = " [충돌]" if ep_collision else ""
        print(f"  [{status}] Ep {ep+1:2d}: "
              f"최소거리={min_dist*100:5.1f}cm | "
              f"각도={np.degrees(min_angle):5.1f}° | "
              f"보상={ep_reward:7.1f}{coll}")

    n_succ = int(sum(successes))
    n_coll = int(sum(collisions))
    print(f"\n{'─'*65}")
    print(f"  성공률:        {np.mean(successes)*100:.1f}%  ({n_succ}/{n_episodes})")
    print(f"  충돌률:        {np.mean(collisions)*100:.1f}%  ({n_coll}/{n_episodes})")
    print(f"  평균 최소거리: {np.mean(distances)*100:.2f} cm")
    print(f"  평균 최소각도: {np.degrees(np.mean(angles)):.2f}°")
    print(f"  평균 보상:     {np.mean(ep_rewards):.2f}")
    print(f"{'─'*65}")

    vec_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M1013 SAC+HER 평가 (Torque)")
    parser.add_argument("--model",    type=str,   default="models/torque/best_model")
    parser.add_argument("--episodes", type=int,   default=20)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--speed",    type=float, default=1.0,
                        help="렌더링 속도 배율 (1.0=실시간)")
    parser.add_argument("--obstacles", type=int,  default=None,
                        help="장애물 수 강제 지정 (미지정 시 저장된 curriculum 값 사용)")
    args = parser.parse_args()

    if os.path.exists(args.model + ".zip") or os.path.exists(args.model):
        evaluate(args.model, n_episodes=args.episodes,
                 render=args.render, speed=args.speed,
                 max_obs_override=args.obstacles)
    else:
        print(f"모델 없음: {args.model}")
        print("먼저 python train.py 를 실행하세요.")
