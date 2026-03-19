"""
Doosan M1013 Robot Arm - Reach Task
Custom MuJoCo Gymnasium Environment

목표: 엔드이펙터(TCP)를 랜덤 타겟 위치로 이동
모델: Doosan Robotics M1013 (6-DOF, max reach 1.3m)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class RobotArmBaseEnv(gym.Env):
    """
    M1013 6축 로봇 팔 Reach 태스크 베이스 환경.
    제어 방식별 구현은 서브클래스(RobotArmPositionEnv, RobotArmTorqueEnv)에서 담당.

    Observation (19차원):
        - joint positions (θ):         6
        - joint velocities:            6
        - end-effector pos (world):    3
        - ee → target vector:          3
        - distance to target:          1

    Curriculum: 성공 threshold를 점진적으로 줄임
        - 시작: 30cm (SUCCESS_THRESHOLD_INIT)
        - 성공률 >= 70% → threshold 절반
        - 성공률 <  20% → threshold 2배 (최대 30cm)
        - 최소: 0.1mm (SUCCESS_THRESHOLD_MIN)

    성공 조건: threshold 내에서 5초(250스텝) 연속 유지
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    JOINT_RANGES = np.array([
        [-6.2832, 6.2832],  # joint1: ±360°
        [-6.2832, 6.2832],  # joint2: ±360°
        [-2.7925, 2.7925],  # joint3: ±160°
        [-6.2832, 6.2832],  # joint4: ±360°
        [-6.2832, 6.2832],  # joint5: ±360°
        [-6.2832, 6.2832],  # joint6: ±360°
    ])

    # 1스텝당 최대 관절 이동량 (rad) — position 제어 전용
    # joint1, joint2는 탐색 범위를 넓혀 대형 이동을 장려
    MAX_DELTA_PER_JOINT = np.array([0.10, 0.10, 0.05, 0.05, 0.05, 0.05])

    # 성공 threshold 범위
    SUCCESS_THRESHOLD_INIT = 0.30   # 30cm (시작)
    SUCCESS_THRESHOLD_MIN  = 0.0001 # 0.1mm (목표)
    SUCCESS_THRESHOLD_MAX  = 0.30   # 30cm (상한)

    XML_FILE = None  # 서브클래스에서 지정

    def __init__(self, render_mode=None, max_episode_steps=1500):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        xml_path = os.path.join(os.path.dirname(__file__), self.XML_FILE)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.n_joints = 6
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(self.n_joints)
        ]
        self.joint_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dof_addr  = [self.model.jnt_dofadr[jid]  for jid in self.joint_ids]

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.target_mocap_idx = self.model.body_mocapid[target_body_id]

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation: q×6 + qvel×6 + ee(3) + vec(3) + dist(1) = 19
        obs_dim = self.n_joints + self.n_joints + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_pos = np.zeros(3)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(self.n_joints)
        self._no_progress_steps = 0
        self._success_steps = 0  # threshold 내 유지 스텝 수 (5초 = 250스텝)
        self.success_threshold = self.SUCCESS_THRESHOLD_INIT
        self._renderer = None

        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ── helpers ──────────────────────────────────────────────────────

    def set_success_threshold(self, threshold: float):
        self.success_threshold = float(np.clip(
            threshold, self.SUCCESS_THRESHOLD_MIN, self.SUCCESS_THRESHOLD_MAX
        ))

    def _sample_target(self) -> np.ndarray:
        """랜덤 관절 각도로 FK를 계산해 도달 가능한 임의의 타겟 위치를 반환.
        현재 qpos를 저장/복원하므로 시작 자세에 영향 없음."""
        saved_qpos = self.data.qpos.copy()
        for i, addr in enumerate(self.joint_qpos_addr):
            self.data.qpos[addr] = np.random.uniform(
                self.JOINT_RANGES[i, 0], self.JOINT_RANGES[i, 1]
            )
        mujoco.mj_forward(self.model, self.data)
        target = self._get_ee_pos().copy()

        # 시작 qpos 복원 (mj_forward는 호출자가 담당)
        self.data.qpos[:] = saved_qpos
        return target

    def _set_target(self, pos: np.ndarray):
        self._target_pos = pos.copy()
        self.data.mocap_pos[self.target_mocap_idx] = pos

    def _get_joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[addr] for addr in self.joint_qpos_addr])

    def _get_joint_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[addr] for addr in self.joint_dof_addr])

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_obs(self) -> np.ndarray:
        qpos = self._get_joint_pos()
        qvel = self._get_joint_vel()
        ee_pos = self._get_ee_pos()
        vec = self._target_pos - ee_pos
        dist = np.linalg.norm(vec)

        return np.concatenate([
            qpos,
            qvel,
            ee_pos,
            vec,
            [dist],
        ]).astype(np.float32)

    # 성공 판정: threshold 내 N스텝 유지
    # hold 시간은 curriculum 진행에 따라 자동 증가:
    #   30cm → 25스텝(0.5s), 15cm → 50스텝(1s), ... 최대 250스텝(5s)
    SUCCESS_HOLD_MAX = 250

    def _required_hold_steps(self) -> int:
        """현재 threshold에 따른 필요 hold 스텝 수 반환.
        threshold 절반마다 2배 증가: 30cm→5스텝(0.1s), 15cm→10s, ..., 최대 250스텝(5s)"""
        return min(
            self.SUCCESS_HOLD_MAX,
            max(5, int(5 * (self.SUCCESS_THRESHOLD_MAX / self.success_threshold)))
        )

    def _compute_reward(self, ee_pos: np.ndarray, action: np.ndarray) -> tuple[float, bool, bool]:
        dist = float(np.linalg.norm(self._target_pos - ee_pos))
        in_threshold = dist < self.success_threshold

        # 1. 접근 보상: 거리가 줄어들수록 즉각 양의 보상
        reward = (self._prev_dist - dist) * 10.0

        # 2. 근접 보너스: 가까울수록 지속적으로 보상 (dense signal)
        reward += float(np.exp(-dist / 0.3))

        # 3. 성공 유지 카운터 + 최종 성공 보너스
        if in_threshold:
            self._success_steps += 1
        else:
            self._success_steps = 0
        success = self._success_steps >= self._required_hold_steps()
        if success:
            reward += 50.0  # hold 달성 보너스

        # 4. 생존 페널티
        reward -= 0.01

        # 5. 관절 한계 근접 패널티: 각 관절이 min/max에 가까울수록 페널티
        qpos = self._get_joint_pos()
        for i in range(self.n_joints):
            dist_to_limit = min(
                qpos[i] - self.JOINT_RANGES[i, 0],
                self.JOINT_RANGES[i, 1] - qpos[i],
            )
            reward -= 0.01 * float(np.exp(-dist_to_limit / 0.3))

        # 6. no-progress 감지 (threshold 밖에서만 카운트)
        improvement = self._prev_dist - dist
        if not in_threshold and improvement < 0.001:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0

        self._prev_dist = dist
        no_progress_truncate = self._no_progress_steps >= 150
        return reward, success, no_progress_truncate

    # ── Gym API ──────────────────────────────────────────────────────

    def _reset_common(self):
        """reset()의 공통 부분: 완전 임의 시작 자세 + 임의 타겟."""
        mujoco.mj_resetData(self.model, self.data)

        # 완전 임의 시작 관절 각도 (전체 범위)
        for i, addr in enumerate(self.joint_qpos_addr):
            self.data.qpos[addr] = np.random.uniform(
                self.JOINT_RANGES[i, 0], self.JOINT_RANGES[i, 1]
            )
        mujoco.mj_forward(self.model, self.data)

        # FK 기반 임의 타겟 샘플링 (qpos 저장/복원 포함)
        target = self._sample_target()
        self._set_target(target)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self._target_pos - self._get_ee_pos()))
        self._prev_action = np.zeros(self.n_joints)
        self._no_progress_steps = 0
        self._success_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        elif self.render_mode == "human":
            try:
                if not hasattr(self, "_viewer") or self._viewer is None:
                    import mujoco.viewer
                    self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.sync()
            except Exception:
                pass

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class RobotArmPositionEnv(RobotArmBaseEnv):
    """
    Position 제어 환경.
    Action: 각 관절의 Δθ ([-1,1] → [-MAX_DELTA, MAX_DELTA] rad)
    MuJoCo position 액추에이터가 내부적으로 PD 제어 수행.
    """

    XML_FILE = "m1013_position.xml"

    def __init__(self, render_mode=None, max_episode_steps=1500):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self._commanded_angles = np.zeros(self.n_joints)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_common()

        self._commanded_angles = self._get_joint_pos().copy()
        self.data.ctrl[:] = self._commanded_angles

        return self._get_obs(), {"target_pos": self._target_pos.copy(), "ee_pos": self._get_ee_pos()}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        self._commanded_angles = np.clip(
            self._commanded_angles + action * self.MAX_DELTA_PER_JOINT,
            self.JOINT_RANGES[:, 0],
            self.JOINT_RANGES[:, 1],
        )
        self.data.ctrl[:] = self._commanded_angles

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        reward, success, no_progress = self._compute_reward(ee_pos, action)
        obs = self._get_obs()

        self._step_count += 1
        dist = float(np.linalg.norm(self._target_pos - ee_pos))
        terminated = success
        truncated = self._step_count >= self.max_episode_steps or no_progress

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {
            "distance": dist, "success": success, "step": self._step_count,
        }


class RobotArmTorqueEnv(RobotArmBaseEnv):
    """
    Torque 제어 환경.
    Action: 각 관절의 정규화 토크 [-1, 1] (MuJoCo motor gear 값으로 실제 Nm 변환)
    """

    XML_FILE = "m1013_torque.xml"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_common()

        return self._get_obs(), {"target_pos": self._target_pos.copy(), "ee_pos": self._get_ee_pos()}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        self.data.ctrl[:] = action

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        reward, success, no_progress = self._compute_reward(ee_pos, action)
        obs = self._get_obs()

        self._step_count += 1
        dist = float(np.linalg.norm(self._target_pos - ee_pos))
        terminated = success
        truncated = self._step_count >= self.max_episode_steps or no_progress

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {
            "distance": dist, "success": success, "step": self._step_count,
        }


gym.register(
    id="M1013Reach-v0",
    entry_point="robot_env:RobotArmPositionEnv",
    max_episode_steps=1500,
)

gym.register(
    id="M1013Reach-Torque-v0",
    entry_point="robot_env:RobotArmTorqueEnv",
    max_episode_steps=1500,
)
