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


class RobotArmEnv(gym.Env):
    """
    M1013 6축 로봇 팔 Reach 태스크 환경.

    Observation (25차원):
        - joint positions (sin/cos):  12  (sin+cos of 6 joints)
        - joint velocities:            6
        - end-effector pos (world):    3
        - ee → target vector:          3
        - distance to target:          1

    Action (6차원):
        - 각 관절 normalized torque [-1, 1]

    Reward:
        - Dense: -distance + proximity bonus
        - Sparse: +10 on success (<5cm)
        - Penalty: joint velocity smoothness
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # M1013 joint limits (radians)
    JOINT_RANGES = np.array([
        [-6.2832, 6.2832],  # joint1: ±360°
        [-6.2832, 6.2832],  # joint2: ±360°
        [-2.7925, 2.7925],  # joint3: ±160°
        [-6.2832, 6.2832],  # joint4: ±360°
        [-6.2832, 6.2832],  # joint5: ±360°
        [-6.2832, 6.2832],  # joint6: ±360°
    ])

    # Practical init range (fraction of full range)
    INIT_RANGE_FRAC = 0.15

    # M1013 workspace target bounds
    TARGET_BOUNDS = {
        "x": (-0.65, 0.65),
        "y": (-0.65, 0.65),
        "z": (0.15, 1.15),
    }

    SUCCESS_THRESHOLD = 0.05  # 5cm

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # MuJoCo 모델 로드
        xml_path = os.path.join(os.path.dirname(__file__), "m1013.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 관절 인덱스
        self.n_joints = 6
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(self.n_joints)
        ]
        self.joint_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dof_addr  = [self.model.jnt_dofadr[jid]  for jid in self.joint_ids]

        # 사이트/바디 인덱스
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.target_free_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free"
        )
        self.target_qpos_addr = self.model.jnt_qposadr[self.target_free_id]

        # Action space: 6 normalized torques
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation space: sin/cos(q) × 6 + qvel × 6 + ee(3) + vec(3) + dist(1)
        obs_dim = self.n_joints * 2 + self.n_joints + 3 + 3 + 1  # 25
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_pos = np.zeros(3)
        self._prev_action = np.zeros(self.n_joints)
        self._renderer = None

        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ── helpers ──────────────────────────────────────────────────────

    def _sample_target(self) -> np.ndarray:
        """M1013 작업 공간 내 랜덤 타겟."""
        return np.array([
            np.random.uniform(*self.TARGET_BOUNDS["x"]),
            np.random.uniform(*self.TARGET_BOUNDS["y"]),
            np.random.uniform(*self.TARGET_BOUNDS["z"]),
        ])

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

        obs = np.concatenate([
            np.sin(qpos),   # 6  (joint angle representation without discontinuity)
            np.cos(qpos),   # 6
            qvel,           # 6
            ee_pos,         # 3
            vec,            # 3
            [dist],         # 1
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, ee_pos: np.ndarray, action: np.ndarray) -> tuple[float, bool]:
        vec = self._target_pos - ee_pos
        dist = float(np.linalg.norm(vec))

        # 1. Dense distance reward (exponential shaping)
        reward = -dist
        reward += np.exp(-5.0 * dist) * 0.5   # proximity bonus

        # 2. Success
        success = dist < self.SUCCESS_THRESHOLD
        if success:
            reward += 10.0

        # 3. Action smoothness (avoid jerky motion)
        reward -= 0.002 * float(np.sum(action ** 2))

        # 4. Joint velocity penalty
        qvel = self._get_joint_vel()
        reward -= 0.001 * float(np.sum(qvel ** 2))

        return reward, success

    # ── Gym API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 초기 관절 각도: home 근처에서 소폭 랜덤화
        for i, addr in enumerate(self.joint_qpos_addr):
            lo = self.JOINT_RANGES[i, 0] * self.INIT_RANGE_FRAC
            hi = self.JOINT_RANGES[i, 1] * self.INIT_RANGE_FRAC
            self.data.qpos[addr] = np.random.uniform(lo, hi)

        # 타겟 배치 및 forward kinematics
        self._set_target(self._sample_target())
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(self.n_joints)

        obs = self._get_obs()
        info = {
            "target_pos": self._target_pos.copy(),
            "ee_pos": self._get_ee_pos(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        # 1 env step = 10 sim steps (0.02s, 50Hz control)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        reward, success = self._compute_reward(ee_pos, action)
        obs = self._get_obs()

        self._step_count += 1
        self._prev_action = action.copy()

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


# Gymnasium 등록
gym.register(
    id="M1013Reach-v0",
    entry_point="robot_env:RobotArmEnv",
    max_episode_steps=500,
)
