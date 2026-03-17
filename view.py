"""
M1013 MuJoCo 뷰어 예제
각 관절에 사인파 모션을 줘서 로봇이 움직이는 걸 확인합니다.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = "m1013.xml"

# 관절별 사인파 파라미터 (amplitude_rad, frequency_hz, phase_offset_rad)
JOINT_MOTIONS = [
    (0.8,  0.3, 0.0),   # joint1: base rotation
    (0.6,  0.2, 0.5),   # joint2: shoulder
    (0.5,  0.25, 1.0),  # joint3: elbow
    (1.0,  0.4, 0.3),   # joint4: forearm roll
    (0.6,  0.3, 0.8),   # joint5: wrist pitch
    (1.2,  0.5, 1.5),   # joint6: wrist roll
]


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    print("M1013 뷰어 시작")
    print("  Space:  시뮬레이션 일시정지/재시작")
    print("  Ctrl+C: 종료\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        dt = model.opt.timestep  # 0.002s

        while viewer.is_running():
            # 각 관절에 사인파 위치 제어
            for i, (amp, freq, phase) in enumerate(JOINT_MOTIONS):
                target = amp * np.sin(2 * np.pi * freq * t + phase)
                data.ctrl[i] = np.clip(target, -1.0, 1.0)

            mujoco.mj_step(model, data)
            viewer.sync()

            t += dt
            time.sleep(dt)


if __name__ == "__main__":
    main()
