"""
학습 중 실시간 모니터링 스크립트

별도 터미널에서 실행하면 best_model.zip이 갱신될 때마다
자동으로 모델을 불러와 MuJoCo 뷰어에서 에이전트 동작을 보여줍니다.

사용법:
    .venv/bin/python monitor.py
    .venv/bin/python monitor.py --model models/best_model --interval 30
"""

import argparse
import os
import time
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
import robot_env  # 환경 등록


def run_episode(model, env, viewer, data, mj_model):
    """에이전트로 에피소드 1회 실행 후 결과 반환."""
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    min_dist = float("inf")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        min_dist = min(min_dist, info["distance"])

        # 뷰어 동기화
        viewer.sync()
        time.sleep(0.02)  # 실제 속도(50Hz)로 재생

        if not viewer.is_running():
            return None

    return {
        "success": info.get("success", False),
        "min_dist": min_dist,
        "reward": total_reward,
    }


def main(model_path: str, check_interval: int):
    print(f"모니터링 시작: {model_path}.zip")
    print(f"모델 갱신 확인 주기: {check_interval}초\n")

    # 환경 & MuJoCo 모델 (뷰어용)
    env = gym.make("M1013Reach-v0", render_mode="human")
    env.set_curriculum_level(1.0)  # 모니터링은 풀 범위로

    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data

    last_mtime = 0
    current_model = None
    ep_count = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth  = 150
        viewer.cam.elevation = -25
        viewer.cam.distance  = 3.0

        print("뷰어 열림. 모델 파일 대기 중...\n")

        while viewer.is_running():
            zip_path = model_path + ".zip"

            # 모델 파일 변경 확인
            if os.path.exists(zip_path):
                mtime = os.path.getmtime(zip_path)
                if mtime != last_mtime:
                    try:
                        current_model = PPO.load(model_path)
                        last_mtime = mtime
                        print(f"  [모델 갱신] {time.strftime('%H:%M:%S')} → {zip_path}")
                    except Exception as e:
                        print(f"  [로드 실패] {e}")

            if current_model is None:
                viewer.sync()
                time.sleep(1.0)
                continue

            # 에피소드 실행
            ep_count += 1
            result = run_episode(current_model, env, viewer, mj_data, mj_model)

            if result is None:
                break  # 뷰어 종료

            status = "✓" if result["success"] else "✗"
            print(
                f"  [{status}] Ep {ep_count:4d} | "
                f"최소거리: {result['min_dist']*100:5.1f}cm | "
                f"보상: {result['reward']:7.1f}"
            )

            # 다음 에피소드 전 잠시 대기 (모델 갱신 확인)
            for _ in range(check_interval):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1.0)

    env.close()
    print("\n모니터링 종료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model")
    parser.add_argument("--interval", type=int, default=10,
                        help="에피소드 사이 대기 시간(초), 이 동안 모델 갱신 확인")
    args = parser.parse_args()

    main(args.model, args.interval)
