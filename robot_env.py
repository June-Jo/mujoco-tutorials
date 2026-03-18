"""
Doosan M1013 Robot Arm - Reach Task
Custom MuJoCo Gymnasium Environment

목표: 엔드이펙터(TCP)를 랜덤 타겟 위치로 이동
모델: Doosan Robotics M1013 (6-DOF, max reach 1.3m)
제어: Position control, action = Δθ (delta joint angle)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class RobotArmEnv(gym.Env):
    """
    M1013 6축 로봇 팔 Reach 태스크 환경.

    Action (6차원):
        - 각 관절의 Δθ (정규화된 [-1, 1] → 실제 [-MAX_DELTA, MAX_DELTA] rad)
        - position 액추에이터가 내부적으로 PD 제어

    Observation (19차원):
        - joint positions (θ):         6
        - joint velocities:            6
        - end-effector pos (world):    3
        - ee → target vector:          3
        - distance to target:          1
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

    # 1스텝당 최대 관절 이동량 (rad) — 50Hz × 0.05 = 2.5 rad/s max
    MAX_DELTA = 0.05

    SUCCESS_THRESHOLD = 0.05  # 5cm

    # curriculum: EE 기준 반경 (level 0→1 선형 보간)
    CURRICULUM_MIN_RADIUS = 0.08   # level=0: 8cm
    CURRICULUM_MAX_RADIUS = 2.6    # level=1: 2×1.3m (M1013 max reach)

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        xml_path = os.path.join(os.path.dirname(__file__), "m1013.xml")
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
        self.target_free_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free"
        )
        self.target_qpos_addr = self.model.jnt_qposadr[self.target_free_id]

        # Action: Δθ per joint, normalized [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation: q×6 + qvel×6 + ee(3) + vec(3) + dist(1)
        obs_dim = self.n_joints + self.n_joints + 3 + 3 + 1  # 19
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_pos = np.zeros(3)
        self._prev_dist = 0.0
        self._prev_action = np.zeros(self.n_joints)
        self.curriculum_level = 0.0  # 0.0 = 쉬움, 1.0 = 최대 범위
        # 현재 position 명령값 (position 액추에이터의 ctrl)
        self._commanded_angles = np.zeros(self.n_joints)
        self._renderer = None

        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ── helpers ──────────────────────────────────────────────────────

    def set_curriculum_level(self, level: float):
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    def _sample_target(self) -> np.ndarray:
        """curriculum_level에 따라 EE 근처 ~ 전체 작업공간에서 샘플링."""
        ee_pos = self._get_ee_pos()
        max_r = self.CURRICULUM_MIN_RADIUS + (
            self.CURRICULUM_MAX_RADIUS - self.CURRICULUM_MIN_RADIUS
        ) * self.curriculum_level

        for _ in range(30):
            r = np.random.uniform(self.SUCCESS_THRESHOLD, max_r)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            offset = r * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            target = ee_pos + offset
            if 0.05 < target[2] < 1.5:
                return target

        return ee_pos + np.array([0.0, 0.0, max_r])

    def _set_target(self, pos: np.ndarray):
        self._target_pos = pos.copy()
        self.data.qpos[self.target_qpos_addr:self.target_qpos_addr + 3] = pos
        self.data.qpos[self.target_qpos_addr + 3] = 1.0
        self.data.qpos[self.target_qpos_addr + 4:self.target_qpos_addr + 7] = 0.0

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

    def _compute_reward(self, ee_pos: np.ndarray, action: np.ndarray) -> tuple[float, bool]:
        dist = float(np.linalg.norm(self._target_pos - ee_pos))
        success = dist < self.SUCCESS_THRESHOLD

        # 1. Potential shaping: 다가갈수록 즉각 양의 보상, 멀어지면 음의 보상
        #    스케일: 0.05rad/step × 50Hz = 약 2.5cm/step 이동 → ~0.025/step 정도
        reward = (self._prev_dist - dist) * 10.0

        # 2. 성공 보너스 (1회성)
        if success:
            reward += 10.0

        # 3. 생존 페널티: 빨리 끝낼수록 유리
        reward -= 0.01

        # 4. Action smoothness: 연속 액션 변화량 페널티
        reward -= 0.001 * float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action.copy()

        self._prev_dist = dist
        return reward, success

    # ── Gym API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 초기 관절 각도 랜덤화: curriculum_level에 따라 전체 범위 비례 확대
        for i, addr in enumerate(self.joint_qpos_addr):
            lo = self.JOINT_RANGES[i, 0] * self.curriculum_level
            hi = self.JOINT_RANGES[i, 1] * self.curriculum_level
            self.data.qpos[addr] = np.random.uniform(lo, hi)

        mujoco.mj_forward(self.model, self.data)

        # 초기 commanded angles = 현재 관절 각도 (제자리 유지)
        self._commanded_angles = self._get_joint_pos().copy()
        self.data.ctrl[:] = self._commanded_angles

        self._set_target(self._sample_target())
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self._target_pos - self._get_ee_pos()))
        self._prev_action = np.zeros(self.n_joints)

        obs = self._get_obs()
        info = {
            "target_pos": self._target_pos.copy(),
            "ee_pos": self._get_ee_pos(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Δθ 적용: commanded_angles 업데이트 후 관절 범위로 클리핑
        self._commanded_angles = np.clip(
            self._commanded_angles + action * self.MAX_DELTA,
            self.JOINT_RANGES[:, 0],
            self.JOINT_RANGES[:, 1],
        )
        self.data.ctrl[:] = self._commanded_angles

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        reward, success = self._compute_reward(ee_pos, action)
        obs = self._get_obs()

        self._step_count += 1
        dist = float(np.linalg.norm(self._target_pos - ee_pos))
        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "distance": dist,
            "success": success,
            "step": self._step_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

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


gym.register(
    id="M1013Reach-v0",
    entry_point="robot_env:RobotArmEnv",
    max_episode_steps=500,
)
