"""
Doosan M1013 Robot Arm - Reach + Orientation
SAC 환경 (Torque 제어 전용)

Observation (flat Box, 33D):
  qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7)
  + target_pos(3) + target_quat(4)

Curriculum (3 variables):
  - pos_threshold:  30cm → 0.1mm
  - ori_threshold:  45° → 0.1°
  - init_range:     ±29° → ±180°
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


# 자가충돌 체크용 로봇 링크 바디명
_ROBOT_BODY_NAMES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]

# 인접 링크 쌍 (조인트로 연결 — 항상 접촉 가능하므로 자가충돌에서 제외)
_ADJACENT_BODY_PAIRS = [
    ("base_link", "link1"),
    ("link1",     "link2"),
    ("link2",     "link3"),
    ("link3",     "link4"),
    ("link4",     "link5"),
    ("link5",     "link6"),
]

SELF_COLLISION_PENALTY = 5.0


# ── 환경 클래스 ───────────────────────────────────────────────────────────

class M1013Env(gym.Env):
    """
    Doosan M1013 6-DOF 로봇 팔 Reach+Orientation 환경 (SAC).

    Observation: flat 33D Box
      qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7)
      + target_pos(3) + target_quat(4)

    Curriculum: pos/ori threshold + init_range 자동 조정
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    JOINT_RANGES = np.array([
        [-6.2832, 6.2832],   # joint1: ±360°
        [-6.2832, 6.2832],   # joint2: ±360°
        [-2.7925, 2.7925],   # joint3: ±160°
        [-6.2832, 6.2832],   # joint4: ±360°
        [-6.2832, 6.2832],   # joint5: ±360°
        [-6.2832, 6.2832],   # joint6: ±360°
    ])
    MAX_QVEL = np.array([2.094, 2.094, 3.142, 3.927, 3.927, 3.927])

    SUCCESS_THRESHOLD_INIT = 0.30
    SUCCESS_THRESHOLD_MIN  = 0.0001   # 0.1mm
    SUCCESS_THRESHOLD_MAX  = 0.30

    ORI_THRESHOLD_INIT = np.pi / 4    # 45°
    ORI_THRESHOLD_MIN  = np.pi / 1800 # 0.1°
    ORI_THRESHOLD_MAX  = np.pi / 4

    INIT_RANGE_INIT = 0.5             # ±0.5 rad (≈ ±29°)
    INIT_RANGE_MAX  = np.pi           # ±π rad  (≈ ±180°)

    SUCCESS_HOLD_STEPS = 5
    XML_FILE = "m1013.xml"

    # flat obs 구성: [qpos(6), qvel(6), ee_pos(3), ee_quat(4), ee_vel(7), target_pos(3), target_quat(4)]
    OBS_DIM = 6 + 6 + 3 + 4 + 7 + 3 + 4  # = 33

    def __init__(self, render_mode=None, max_episode_steps=200):
        super().__init__()
        self.render_mode       = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count       = 0

        xml_path = os.path.join(os.path.dirname(__file__), self.XML_FILE)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # 관절 인덱스
        self.n_joints = 6
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            for i in range(self.n_joints)
        ]
        self.joint_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dof_addr  = [self.model.jnt_dofadr[jid]  for jid in self.joint_ids]

        # EE site
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # Target mocap
        target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_idx = self.model.body_mocapid[target_bid]

        # 자가충돌 체크: 로봇 body ID 집합 + 제외할 인접 쌍
        self._robot_body_ids = set()
        for name in _ROBOT_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._robot_body_ids.add(bid)

        self._adjacent_body_pairs = set()
        for n1, n2 in _ADJACENT_BODY_PAIRS:
            b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n1)
            b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n2)
            if b1 >= 0 and b2 >= 0:
                self._adjacent_body_pairs.add((min(b1, b2), max(b1, b2)))

        # Curriculum 상태
        self.success_threshold = self.SUCCESS_THRESHOLD_INIT
        self.ori_threshold     = self.ORI_THRESHOLD_INIT
        self._init_qpos_range  = self.INIT_RANGE_INIT
        self._success_steps    = 0
        self._target_pos       = np.zeros(3)
        self._target_quat      = np.array([1.0, 0.0, 0.0, 0.0])
        self._prev_dist        = 0.0
        self._prev_angle_err   = 0.0

        # Observation / Action space
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ── Curriculum setters ────────────────────────────────────────────

    def set_success_threshold(self, threshold: float, ori_threshold: float = None):
        self.success_threshold = float(np.clip(
            threshold, self.SUCCESS_THRESHOLD_MIN, self.SUCCESS_THRESHOLD_MAX
        ))
        if ori_threshold is not None:
            self.ori_threshold = float(np.clip(
                ori_threshold, self.ORI_THRESHOLD_MIN, self.ORI_THRESHOLD_MAX
            ))

    def set_init_range(self, init_range: float):
        self._init_qpos_range = float(np.clip(
            init_range, self.INIT_RANGE_INIT, self.INIT_RANGE_MAX
        ))

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _get_joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.joint_qpos_addr])

    def _get_joint_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self.joint_dof_addr])

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_ee_quat(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        return quat

    def _get_ee_vel(self) -> np.ndarray:
        """EE 선속도(3D) + 쿼터니언 도함수(4D) = 7D."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        lin_vel = jacp @ self.data.qvel
        ang_vel = jacr @ self.data.qvel
        q = self._get_ee_quat()
        qw, qx, qy, qz = q
        ox, oy, oz = ang_vel
        qdot = 0.5 * np.array([
            -(ox*qx + oy*qy + oz*qz),
             ox*qw + oy*qz - oz*qy,
            -ox*qz + oy*qw + oz*qx,
             ox*qy - oy*qx + oz*qw,
        ])
        return np.concatenate([lin_vel, qdot]).astype(np.float32)

    @staticmethod
    def _angle_between_quats(q1: np.ndarray, q2: np.ndarray) -> float:
        dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
        return 2.0 * np.arccos(dot)

    def _check_self_collision(self) -> bool:
        """비인접 로봇 링크 간 자가충돌 감지 (MuJoCo contact 기반)."""
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 not in self._robot_body_ids or b2 not in self._robot_body_ids:
                continue
            pair = (min(b1, b2), max(b1, b2))
            if pair not in self._adjacent_body_pairs:
                return True
        return False

    def _sample_target_pose(self) -> tuple:
        """FK 기반으로 도달 가능한 타겟 위치+자세를 동시에 샘플링."""
        saved_qpos   = self.data.qpos.copy()
        current_pos  = self._get_ee_pos()
        current_quat = self._get_ee_quat()
        max_dist     = min(1.0, self.success_threshold * 10)

        result_pos  = None
        result_quat = None
        for _ in range(50):
            candidate_qpos = np.array([
                np.random.uniform(self.JOINT_RANGES[i, 0], self.JOINT_RANGES[i, 1])
                for i in range(len(self.joint_qpos_addr))
            ])
            for i, addr in enumerate(self.joint_qpos_addr):
                self.data.qpos[addr] = candidate_qpos[i]
            mujoco.mj_forward(self.model, self.data)
            candidate_pos = self._get_ee_pos().copy()
            if np.linalg.norm(candidate_pos - current_pos) <= max_dist:
                result_pos  = candidate_pos
                result_quat = self._get_ee_quat().copy()
                break

        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)

        if result_pos is None:
            result_pos  = current_pos.copy()
            result_quat = current_quat.copy()

        return result_pos, result_quat

    def _set_target(self, pos: np.ndarray, quat: np.ndarray):
        self._target_pos  = pos.copy()
        self._target_quat = quat.copy()
        self.data.mocap_pos[self.target_mocap_idx]  = pos
        self.data.mocap_quat[self.target_mocap_idx] = quat

    # ── Observation / Reward ─────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """flat 33D 관측 벡터 반환."""
        return np.concatenate([
            self._get_joint_pos(),                              # 6D
            self._get_joint_vel(),                              # 6D
            self._get_ee_pos(),                                 # 3D
            self._get_ee_quat(),                                # 4D
            self._get_ee_vel(),                                 # 7D
            self._target_pos.astype(np.float32),                # 3D
            self._target_quat.astype(np.float32),               # 4D
        ]).astype(np.float32)

    def _step_reward(self, obs: np.ndarray) -> tuple:
        """(reward, success, terminated, dist, angle_err) 반환.

        보상 구성:
          1) 거리 보상:      -(pos_dist + angle_error)
          2) 진행 보상:      (prev_dist - dist) + (prev_angle - angle_err)
          3) 정지 보상:      proximity × exp(-lin_vel × 5)
             proximity = clip(1 - dist / success_threshold, 0, 1)
          4) 성공 보너스:    +10
          5) 자가충돌 패널티: -5 + terminated
        """
        ee_pos  = self._get_ee_pos()
        ee_quat = self._get_ee_quat()

        dist      = float(np.linalg.norm(self._target_pos - ee_pos))
        angle_err = self._angle_between_quats(ee_quat, self._target_quat)

        # 1. 거리 보상
        reward = -(dist + angle_err)

        # 2. 진행 보상
        reward += (self._prev_dist - dist) + (self._prev_angle_err - angle_err)

        # 3. 정지 보상 (목표 반경 1× 이내, 낮은 선속도일수록 높음)
        lin_vel   = float(np.linalg.norm(obs[19:22]))
        proximity = float(np.clip(1.0 - dist / self.success_threshold, 0.0, 1.0))
        reward   += proximity * float(np.exp(-lin_vel * 5.0))

        # prev 갱신
        self._prev_dist      = dist
        self._prev_angle_err = angle_err

        # 성공 판정
        if dist < self.success_threshold and angle_err < self.ori_threshold:
            self._success_steps += 1
        else:
            self._success_steps = 0

        success = self._success_steps >= self.SUCCESS_HOLD_STEPS
        if success:
            reward += 10.0

        # 자가충돌 패널티
        terminated = False
        if self._check_self_collision():
            reward    -= SELF_COLLISION_PENALTY
            terminated = True

        return reward, success, terminated, dist, angle_err

    # ── Gym API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 초기 관절 자세 (curriculum init_range)
        for i, addr in enumerate(self.joint_qpos_addr):
            self.data.qpos[addr] = float(np.clip(
                np.random.uniform(-self._init_qpos_range, self._init_qpos_range),
                self.JOINT_RANGES[i, 0], self.JOINT_RANGES[i, 1],
            ))
        mujoco.mj_forward(self.model, self.data)

        target_pos, target_quat = self._sample_target_pose()
        self._set_target(target_pos, target_quat)

        mujoco.mj_forward(self.model, self.data)
        self._step_count    = 0
        self._success_steps = 0

        obs = self._get_obs()
        self._prev_dist      = float(np.linalg.norm(self._target_pos - self._get_ee_pos()))
        self._prev_angle_err = self._angle_between_quats(self._get_ee_quat(), self._target_quat)

        return obs, {
            "target_pos":  self._target_pos.copy(),
            "target_quat": self._target_quat.copy(),
            "ee_pos":      self._get_ee_pos(),
        }

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # 관절 속도 제한
        for i, addr in enumerate(self.joint_dof_addr):
            self.data.qvel[addr] = float(np.clip(
                self.data.qvel[addr], -self.MAX_QVEL[i], self.MAX_QVEL[i]
            ))

        obs = self._get_obs()
        reward, success, self_collision, dist, angle_err = self._step_reward(obs)

        self._step_count += 1
        terminated = success or self_collision
        truncated  = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {
            "distance":       dist,
            "angle_error":    angle_err,
            "success":        success,
            "self_collision": self_collision,
            "step":           self._step_count,
        }

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


# 하위 호환성 alias
RobotArmBaseEnv = M1013Env

gym.register(
    id="M1013Reach-v0",
    entry_point="robot_env:M1013Env",
    max_episode_steps=200,
)
