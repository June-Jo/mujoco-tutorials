"""
[LEGACY] RobotArmPositionEnv — Position 제어 환경 (더 이상 사용하지 않음)
Torque 제어로 전환됨. 참조용으로만 보존.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class RobotArmPositionEnv(gym.Env):
    XML_FILE = "m1013_position.xml"

    JOINT_RANGES = np.array([
        [-6.2832, 6.2832],
        [-6.2832, 6.2832],
        [-2.7925, 2.7925],
        [-6.2832, 6.2832],
        [-6.2832, 6.2832],
        [-6.2832, 6.2832],
    ])
    MAX_QVEL = np.array([2.094, 2.094, 3.142, 3.927, 3.927, 3.927])
    MAX_DELTA_PER_JOINT = np.array([0.042, 0.042, 0.063, 0.079, 0.079, 0.079])
    SUCCESS_THRESHOLD_INIT = 0.30
    SUCCESS_THRESHOLD_MIN  = 0.0001
    SUCCESS_THRESHOLD_MAX  = 0.30
    ORI_THRESHOLD_INIT = np.pi / 4
    ORI_THRESHOLD_MIN  = np.pi / 1800
    ORI_THRESHOLD_MAX  = np.pi / 4
    INIT_RANGE_INIT = 0.5
    INIT_RANGE_MAX  = np.pi
    SUCCESS_HOLD_STEPS = 5

    def __init__(self, render_mode=None, max_episode_steps=200):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        xml_path = os.path.join(os.path.dirname(__file__), "..", self.XML_FILE)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.n_joints = 6
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(self.n_joints)
        ]
        self.joint_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dof_addr  = [self.model.jnt_dofadr[jid]  for jid in self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_idx = self.model.body_mocapid[target_body_id]

        self._commanded_angles = np.zeros(self.n_joints)
        self.success_threshold = self.SUCCESS_THRESHOLD_INIT
        self.ori_threshold = self.ORI_THRESHOLD_INIT
        self._init_qpos_range = self.INIT_RANGE_INIT
        self._success_steps = 0
        self._target_pos  = np.zeros(3)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        obs_dim = self.n_joints * 2 + 7 + 7
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_joints,), dtype=np.float32)
