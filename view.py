"""
M1013 MuJoCo 뷰어 예제
각 관절에 사인파 모션을 줘서 로봇이 움직이는 걸 확인합니다.

사용법:
    python view.py              # position 제어 (기본)
    python view.py --torque     # torque 제어
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time

# 관절별 사인파 파라미터 (amplitude, frequency_hz, phase_offset_rad)
# position mode: amplitude = 목표 각도 (rad)
# torque mode:   amplitude = 정규화 토크 [-1, 1]
JOINT_MOTIONS_POSITION = [
    (0.8,  0.3, 0.0),   # joint1: base rotation
    (0.6,  0.2, 0.5),   # joint2: shoulder
    (0.5,  0.25, 1.0),  # joint3: elbow
    (1.0,  0.4, 0.3),   # joint4: forearm roll
    (0.6,  0.3, 0.8),   # joint5: wrist pitch
    (1.2,  0.5, 1.5),   # joint6: wrist roll
]

JOINT_MOTIONS_TORQUE = [
    (0.5,  0.3, 0.0),   # joint1
    (0.8,  0.2, 0.5),   # joint2: 중력 영향 크므로 진폭 높임
    (0.6,  0.25, 1.0),  # joint3
    (0.4,  0.4, 0.3),   # joint4
    (0.4,  0.3, 0.8),   # joint5
    (0.3,  0.5, 1.5),   # joint6
]


def main(control_mode: str):
    xml_path = "m1013_position.xml" if control_mode == "position" else "m1013_torque.xml"
    joint_motions = JOINT_MOTIONS_POSITION if control_mode == "position" else JOINT_MOTIONS_TORQUE

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    print(f"M1013 뷰어 시작 ({control_mode} 제어)")
    print("  Space:  시뮬레이션 일시정지/재시작")
    print("  Ctrl+C: 종료\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        dt = model.opt.timestep  # 0.002s

        while viewer.is_running():
            for i, (amp, freq, phase) in enumerate(joint_motions):
                target = amp * np.sin(2 * np.pi * freq * t + phase)
                data.ctrl[i] = np.clip(target, -1.0, 1.0)

            mujoco.mj_step(model, data)
            viewer.sync()

            t += dt
            time.sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torque", action="store_true",
                        help="토크 제어 모드로 실행 (기본값: position 제어)")
    args = parser.parse_args()

    main("torque" if args.torque else "position")
