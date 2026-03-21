"""
Doosan M1013 Robot Arm - Reach + Orientation + Dynamic Obstacle Avoidance
SAC + HER 환경 (Torque 제어 전용)

Observation (GoalEnv Dict):
  - observation:    qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7)
                    + desired_speed(1) = 27D
  - achieved_goal:  ee_pos(3) + ee_quat(4) = 7D
  - desired_goal:   target_pos(3) + target_quat(4) = 7D
  - obstacles:      10 × [cx, cy, cz, p1, p2, p3, type_flag] = 70D (flat)
                    type_flag: 0=sphere, 1=box, 2=capsule, -1=inactive

Curriculum (4 variables):
  - pos_threshold:  30cm → 0.1mm
  - ori_threshold:  45° → 0.1°
  - init_range:     ±29° → ±180°
  - max_obs_count:  0 → 10  (pos_threshold < 10cm 이후 활성화)

Dynamic obstacles:
  - 100개 풀 (구 34 + 박스 33 + 캡슐 33)
  - 에피소드마다 0~max_obs_count개 무작위 활성화
  - 활성 장애물은 매 스텝 이동 (경계 도달 시 반사)
  - 충돌(EE가 장애물 내부 진입) 시 패널티 + 에피소드 종료

desired_speed (Future Work):
  - [0, 1] 사이 값을 에피소드마다 랜덤 샘플, observation에 포함
  - 현재는 보상 미반영 — 향후 속도 추적 보상 추가 예정
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


# ── 장애물 설정 ───────────────────────────────────────────────────────────

MAX_OBSTACLES   = 10
OBS_FEAT_DIM    = 7      # 슬롯당 피처: [cx, cy, cz, p1, p2, p3, type_flag]
CONTROL_DT      = 0.002 * 10  # 0.02s per control step (10 sub-steps)

OBS_UNLOCK_THRESHOLD = 0.10   # pos_threshold < 10cm 이후 장애물 커리큘럼 시작
OBS_WS_LOW  = np.array([-0.90, -0.90, 0.15])   # 장애물 활성화 workspace
OBS_WS_HIGH = np.array([ 0.90,  0.90, 1.40])
OBS_MIN_SPEED = 0.05   # m/s
OBS_MAX_SPEED = 0.25   # m/s
COLLISION_PENALTY = 5.0
PARK_POS = np.array([0.0, 0.0, -2.0])  # 비활성 장애물 주차 위치

# 장애물 풀 (100개 형상)

# Sphere: radius (34개, meters)
SPHERE_POOL = np.array([
    0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075,
    0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150,
    0.160, 0.170, 0.180, 0.190, 0.200, 0.220, 0.250, 0.048, 0.058, 0.068,
    0.078, 0.088, 0.098, 0.108,
])  # 34개

# Box: half-sizes [hx, hy, hz] (33개)
BOX_POOL = np.array([
    [0.040, 0.040, 0.040], [0.080, 0.040, 0.040], [0.040, 0.080, 0.040],
    [0.040, 0.040, 0.080], [0.080, 0.080, 0.040], [0.080, 0.040, 0.080],
    [0.040, 0.080, 0.080], [0.080, 0.080, 0.080], [0.120, 0.040, 0.040],
    [0.040, 0.120, 0.040], [0.040, 0.040, 0.120], [0.120, 0.080, 0.040],
    [0.120, 0.040, 0.080], [0.080, 0.120, 0.040], [0.040, 0.120, 0.080],
    [0.080, 0.040, 0.120], [0.040, 0.080, 0.120], [0.120, 0.120, 0.040],
    [0.120, 0.040, 0.120], [0.040, 0.120, 0.120], [0.120, 0.120, 0.080],
    [0.120, 0.080, 0.120], [0.080, 0.120, 0.120], [0.120, 0.120, 0.120],
    [0.160, 0.040, 0.040], [0.040, 0.160, 0.040], [0.040, 0.040, 0.160],
    [0.160, 0.080, 0.040], [0.160, 0.040, 0.080], [0.080, 0.160, 0.040],
    [0.040, 0.160, 0.080], [0.080, 0.040, 0.160], [0.040, 0.080, 0.160],
])  # 33개

# Capsule: [radius, half_cyl_length] (33개)
# MuJoCo capsule size: size[0]=radius, size[1]=half cylinder length (not including caps)
CAPSULE_POOL = np.array([
    [0.030, 0.100], [0.030, 0.150], [0.030, 0.200],
    [0.040, 0.100], [0.040, 0.150], [0.040, 0.200], [0.040, 0.250],
    [0.050, 0.100], [0.050, 0.150], [0.050, 0.200], [0.050, 0.250], [0.050, 0.300],
    [0.060, 0.100], [0.060, 0.150], [0.060, 0.200], [0.060, 0.250], [0.060, 0.300],
    [0.070, 0.150], [0.070, 0.200], [0.070, 0.250], [0.070, 0.300],
    [0.080, 0.150], [0.080, 0.200], [0.080, 0.250], [0.080, 0.300],
    [0.090, 0.200], [0.090, 0.250], [0.090, 0.300],
    [0.100, 0.200], [0.100, 0.250], [0.100, 0.300], [0.100, 0.350], [0.100, 0.400],
])  # 33개

# 100개 통합 장애물 풀 (자유롭게 슬롯에 배정 가능)
OBSTACLE_POOL = (
    [('sphere',  np.array([r]))     for r in SPHERE_POOL] +   # 34개
    [('box',     p.copy())          for p in BOX_POOL]    +   # 33개
    [('capsule', p.copy())          for p in CAPSULE_POOL]    # 33개
)  # 합계 100개

_TYPE_FLAG  = {'sphere': 0.0, 'box': 1.0, 'capsule': 2.0}

# ── 로봇 링크 충돌 체크 설정 ─────────────────────────────────────────────

# 링크 body 이름 및 보수적 반경 (m) — 실제 geom보다 약간 크게 설정
_LINK_BODY_NAMES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
_LINK_RADII      = [      0.09,     0.08,    0.07,    0.07,    0.06,    0.05,    0.04]
_EE_RADIUS       = 0.04

# (idx_a, idx_b, n_mid, radius): 긴 세그먼트 중간 샘플링
# link2(idx=2)→link3(idx=3): 0.62m,  link3(idx=3)→link4(idx=4): 0.559m
_LONG_SEGMENT_DEFS = [(2, 3, 2, 0.07), (3, 4, 2, 0.06)]

# ── desired_speed 보상 설정 ───────────────────────────────────────────────
# desired_speed: 사용자 입력 [0.0, 1.0] → observation에 포함
# meta_speed:    목표 EE 속도 (m/s) = SPEED_META_MIN + desired_speed × (SPEED_META_MAX - SPEED_META_MIN)
#                [0.1, 0.9] m/s — M1013 최대 선속도(1.0 m/s) 기준, 극단값 회피
SPEED_META_MIN    = 0.1   # 목표 EE 속도 하한 (m/s)
SPEED_META_MAX    = 0.9   # 목표 EE 속도 상한 (m/s, M1013 최대 1.0m/s 이하)
SPEED_WEIGHT_MAX  = 0.2   # 속도 보상 가중치 상한 (커리큘럼으로 0→0.2 증가)
SPEED_WEIGHT_STEP = 0.05  # 커리큘럼 업데이트 시 증가량
SPEED_GATE_K      = 2.0   # 거리 게이팅: dist < threshold×K 이하면 속도 보상 감쇠
SPEED_STEPS_MIN   = 150   # max_episode_steps 하한
SPEED_STEPS_MAX   = 500   # max_episode_steps 상한
SPEED_UNLOCK_THRESHOLD = 0.05  # pos_threshold < 5cm 이후 speed reward 커리큘럼 시작
# 공식: max_episode_steps = clip(100 / meta_speed, 150, 500)   [max_dist=1.0m, CONTROL_DT=0.02]
# 슬롯별 활성 geom 색상
_TYPE_RGBA  = {
    'sphere':  np.array([0.85, 0.25, 0.25, 0.75]),
    'box':     np.array([0.25, 0.75, 0.30, 0.75]),
    'capsule': np.array([0.25, 0.45, 0.90, 0.75]),
}


# ── 환경 클래스 ───────────────────────────────────────────────────────────

class M1013Env(gym.Env):
    """
    Doosan M1013 6-DOF 로봇 팔 Reach+Orientation+장애물 회피 환경 (SAC + HER).

    GoalEnv 스타일 Dict Observation:
      observation(27D) + achieved_goal(7D) + desired_goal(7D) + obstacles(70D)

    Curriculum:
      pos/ori threshold + init_range + max_obs_count 자동 조정
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

    def __init__(self, render_mode=None, max_episode_steps=200):
        super().__init__()
        self.render_mode      = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count      = 0

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

        # 로봇 링크 body IDs (충돌 체크용)
        self._link_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _LINK_BODY_NAMES
        ]
        # 긴 세그먼트 중간점 체크: (body_id_a, body_id_b, n_mid, radius)
        self._link_segment_checks = [
            (self._link_body_ids[a], self._link_body_ids[b], n, r)
            for a, b, n, r in _LONG_SEGMENT_DEFS
        ]

        # Obstacle mocap / geom IDs (슬롯 0~9, 각 슬롯에 sphere/box/capsule 3개 geom)
        self._obs_mocap_ids   = []
        self._obs_geom_ids    = []  # list of dict: {'sphere': gid, 'box': gid, 'capsule': gid}
        for i in range(MAX_OBSTACLES):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
            self._obs_mocap_ids.append(int(self.model.body_mocapid[bid]))
            self._obs_geom_ids.append({
                'sphere':  int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}_s")),
                'box':     int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}_b")),
                'capsule': int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}_c")),
            })

        # 장애물 상태
        self.max_obs_count    = 0
        self.fixed_obs_count  = None   # None=랜덤(0~max), int=고정 수
        self._obs_active      = np.zeros(MAX_OBSTACLES, dtype=bool)
        self._obs_vel         = np.zeros((MAX_OBSTACLES, 3))
        self._obs_params      = [None] * MAX_OBSTACLES   # 슬롯별 현재 size params
        self._obs_active_type = [None] * MAX_OBSTACLES   # 슬롯별 현재 활성 타입 str

        # Curriculum 상태
        self.success_threshold  = self.SUCCESS_THRESHOLD_INIT
        self.ori_threshold      = self.ORI_THRESHOLD_INIT
        self._init_qpos_range   = self.INIT_RANGE_INIT
        self._success_steps     = 0
        self._target_pos        = np.zeros(3)
        self._target_quat       = np.array([1.0, 0.0, 0.0, 0.0])

        # Desired speed (future work)
        self._desired_speed = 0.5   # [0.0, 1.0]
        self._meta_speed    = 0.5   # [0.1, 0.9] = 0.1 + desired_speed × 0.8
        self.speed_weight   = 0.0   # 커리큘럼으로 0 → SPEED_WEIGHT_MAX 증가

        # Observation / Action space
        obs_dim = self.n_joints + self.n_joints + 7 + 7 + 1  # 27D
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(obs_dim,),                       dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(7,),                             dtype=np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(7,),                             dtype=np.float32),
            "obstacles":     spaces.Box(-np.inf, np.inf, shape=(MAX_OBSTACLES * OBS_FEAT_DIM,), dtype=np.float32),
        })
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

    def set_max_obs_count(self, max_obs_count: int):
        self.max_obs_count = int(np.clip(max_obs_count, 0, MAX_OBSTACLES))

    def set_speed_weight(self, weight: float):
        self.speed_weight = float(np.clip(weight, 0.0, SPEED_WEIGHT_MAX))

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

    def _sample_target_pose(self) -> tuple:
        """FK 기반으로 도달 가능한 타겟 위치+자세를 동시에 샘플링.

        같은 관절 설정에서 EE 위치와 자세를 함께 가져오므로
        반환값은 항상 실제로 도달 가능한 pose임이 보장됨.
        """
        saved_qpos   = self.data.qpos.copy()
        current_pos  = self._get_ee_pos()
        current_quat = self._get_ee_quat()
        max_dist     = min(1.0, self.success_threshold * 10)

        result_pos  = None
        result_quat = None
        for _ in range(50):
            for i, addr in enumerate(self.joint_qpos_addr):
                self.data.qpos[addr] = np.random.uniform(
                    self.JOINT_RANGES[i, 0], self.JOINT_RANGES[i, 1]
                )
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

    # ── 장애물 관리 ──────────────────────────────────────────────────

    def _set_slot_rgba(self, i: int, active_type: str):
        """슬롯 i의 활성 geom을 표시하고 나머지 2개를 투명하게 설정."""
        geoms = self._obs_geom_ids[i]
        for t, gid in geoms.items():
            if t == active_type:
                self.model.geom_rgba[gid] = _TYPE_RGBA[t]
            else:
                self.model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]

    def _hide_slot(self, i: int):
        """슬롯 i의 모든 geom을 투명하게 설정."""
        for gid in self._obs_geom_ids[i].values():
            self.model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]

    def _reset_obstacles(self):
        """에피소드 시작 시 100개 풀에서 무작위 샘플링 후 장애물 배치."""
        if self.fixed_obs_count is not None:
            n_active = int(np.clip(self.fixed_obs_count, 0, MAX_OBSTACLES))
        else:
            n_active = np.random.randint(0, self.max_obs_count + 1)
        # 100개 풀에서 n_active개 비복원 추출
        sampled_indices = np.random.choice(len(OBSTACLE_POOL), size=n_active, replace=False)
        active_set = set(range(n_active))

        for i in range(MAX_OBSTACLES):
            if i < n_active:
                pool_type, pool_params = OBSTACLE_POOL[sampled_indices[i]]

                pos       = np.random.uniform(OBS_WS_LOW, OBS_WS_HIGH)
                speed     = np.random.uniform(OBS_MIN_SPEED, OBS_MAX_SPEED)
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)

                self.data.mocap_pos[self._obs_mocap_ids[i]]  = pos
                self.data.mocap_quat[self._obs_mocap_ids[i]] = [1, 0, 0, 0]

                # 활성 geom 크기 설정
                gid = self._obs_geom_ids[i][pool_type]
                if pool_type == 'sphere':
                    self.model.geom_size[gid, 0] = float(pool_params[0])
                elif pool_type == 'box':
                    self.model.geom_size[gid, :3] = pool_params
                else:  # capsule
                    self.model.geom_size[gid, 0] = pool_params[0]
                    self.model.geom_size[gid, 1] = pool_params[1]

                # rgba: 활성 타입만 표시
                self._set_slot_rgba(i, pool_type)

                self._obs_active[i]      = True
                self._obs_vel[i]         = direction * speed
                self._obs_params[i]      = pool_params.copy()
                self._obs_active_type[i] = pool_type
            else:
                self.data.mocap_pos[self._obs_mocap_ids[i]] = PARK_POS
                self._hide_slot(i)
                self._obs_active[i]      = False
                self._obs_vel[i]         = 0.0
                self._obs_params[i]      = None
                self._obs_active_type[i] = None

    def _step_obstacles(self):
        """매 스텝 활성 장애물 위치 업데이트 (반사 경계)."""
        for i in range(MAX_OBSTACLES):
            if not self._obs_active[i]:
                continue

            pos = self.data.mocap_pos[self._obs_mocap_ids[i]].copy()
            vel = self._obs_vel[i].copy()
            new_pos = pos + vel * CONTROL_DT

            for dim in range(3):
                if new_pos[dim] < OBS_WS_LOW[dim]:
                    new_pos[dim] = OBS_WS_LOW[dim]
                    vel[dim]     = abs(vel[dim])
                elif new_pos[dim] > OBS_WS_HIGH[dim]:
                    new_pos[dim] = OBS_WS_HIGH[dim]
                    vel[dim]     = -abs(vel[dim])

            self._obs_vel[i] = vel
            self.data.mocap_pos[self._obs_mocap_ids[i]] = new_pos

    def _get_obs_obstacles(self) -> np.ndarray:
        """장애물 관측 벡터 반환 (70D flat). 비활성 슬롯: type=-1, z=-2."""
        feat = np.zeros((MAX_OBSTACLES, OBS_FEAT_DIM), dtype=np.float32)
        feat[:, 2] = PARK_POS[2]  # z = -2 for inactive
        feat[:, 6] = -1.0         # type = -1 for inactive

        for i in range(MAX_OBSTACLES):
            if not self._obs_active[i]:
                continue
            pos       = self.data.mocap_pos[self._obs_mocap_ids[i]]
            slot_type = self._obs_active_type[i]
            params    = self._obs_params[i]
            type_flag = _TYPE_FLAG[slot_type]

            if slot_type == 'sphere':
                feat[i] = [pos[0], pos[1], pos[2], float(params[0]), 0.0, 0.0, type_flag]
            elif slot_type == 'box':
                feat[i] = [pos[0], pos[1], pos[2], params[0], params[1], params[2], type_flag]
            else:  # capsule
                feat[i] = [pos[0], pos[1], pos[2], params[0], params[1], 0.0, type_flag]

        return feat.flatten()

    @staticmethod
    def _dist_point_to_obs(pt: np.ndarray, obs_pos: np.ndarray,
                           slot_type: str, params: np.ndarray) -> float:
        """점 pt 에서 장애물 표면까지의 부호 거리.
        sphere/capsule: 음수=내부, 양수=외부
        box: 0=내부 또는 표면, 양수=외부 (AABB external distance)
        """
        if slot_type == 'sphere':
            return np.linalg.norm(pt - obs_pos) - float(params[0])
        elif slot_type == 'box':
            hx, hy, hz = params
            dx = max(0.0, abs(pt[0] - obs_pos[0]) - hx)
            dy = max(0.0, abs(pt[1] - obs_pos[1]) - hy)
            dz = max(0.0, abs(pt[2] - obs_pos[2]) - hz)
            return np.sqrt(dx*dx + dy*dy + dz*dz)
        else:  # capsule (z-axis aligned)
            radius, half_len = params
            dp = pt - obs_pos
            t  = float(np.clip(dp[2], -half_len, half_len))
            return np.sqrt(dp[0]**2 + dp[1]**2 + (dp[2] - t)**2) - radius

    def _check_collision(self, ee_pos: np.ndarray) -> bool:
        """로봇 링크(+EE) 전체가 활성 장애물과 겹치면 True.

        체크 포인트:
          - 7개 링크 body 위치 (보수적 반경 적용)
          - EE site 위치
          - link2→link3, link3→link4 긴 세그먼트 중간점 2개씩
        합계 ~13개 구형 근사 포인트.
        """
        # 체크 포인트 구성: (world_pos, link_radius)
        check_pts = [
            (self.data.xpos[bid], r)
            for bid, r in zip(self._link_body_ids, _LINK_RADII)
        ]
        check_pts.append((ee_pos, _EE_RADIUS))
        for bid_a, bid_b, n_mid, r in self._link_segment_checks:
            pa = self.data.xpos[bid_a]
            pb = self.data.xpos[bid_b]
            for k in range(1, n_mid + 1):
                t = k / (n_mid + 1)
                check_pts.append((pa + t * (pb - pa), r))

        for i in range(MAX_OBSTACLES):
            if not self._obs_active[i]:
                continue
            obs_pos   = self.data.mocap_pos[self._obs_mocap_ids[i]]
            slot_type = self._obs_active_type[i]
            params    = self._obs_params[i]

            for pt, link_r in check_pts:
                if self._dist_point_to_obs(pt, obs_pos, slot_type, params) < link_r:
                    return True
        return False

    # ── Observation / Reward ─────────────────────────────────────────

    def _get_obs(self) -> dict:
        qpos    = self._get_joint_pos()
        qvel    = self._get_joint_vel()
        ee_pos  = self._get_ee_pos()
        ee_quat = self._get_ee_quat()
        ee_vel  = self._get_ee_vel()
        robot_obs = np.concatenate([
            qpos, qvel, ee_pos, ee_quat, ee_vel,
            [self._desired_speed],
        ]).astype(np.float32)

        return {
            "observation":   robot_obs,
            "achieved_goal": np.concatenate([ee_pos, ee_quat]).astype(np.float32),
            "desired_goal":  np.concatenate([self._target_pos, self._target_quat]).astype(np.float32),
            "obstacles":     self._get_obs_obstacles(),
        }

    def compute_reward(self, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, info) -> np.ndarray:
        """HER 재레이블링용 보상 함수 (vectorized). step과 동일한 함수 사용."""
        pos_dist = np.linalg.norm(
            achieved_goal[..., :3] - desired_goal[..., :3], axis=-1
        )
        q_a = achieved_goal[..., 3:]
        q_d = desired_goal[..., 3:]
        norm_a = np.linalg.norm(q_a, axis=-1, keepdims=True)
        norm_d = np.linalg.norm(q_d, axis=-1, keepdims=True)
        q_a = q_a / np.where(norm_a > 1e-8, norm_a, 1.0)
        q_d = q_d / np.where(norm_d > 1e-8, norm_d, 1.0)
        dot   = np.clip(np.abs(np.sum(q_a * q_d, axis=-1)), 0.0, 1.0)
        angle = 2.0 * np.arccos(dot)
        return -(pos_dist + 0.3 * angle).astype(np.float32)

    def _step_reward(self, obs: dict) -> tuple:
        """(reward, success, terminated, dist, angle_err) 반환."""
        achieved = obs["achieved_goal"]
        desired  = obs["desired_goal"]

        reward = float(self.compute_reward(
            achieved[np.newaxis], desired[np.newaxis], {}
        )[0])

        ee_pos    = achieved[:3]
        ee_quat   = achieved[3:]
        dist      = float(np.linalg.norm(self._target_pos - ee_pos))
        angle_err = self._angle_between_quats(ee_quat, self._target_quat)

        if dist < self.success_threshold and angle_err < self.ori_threshold:
            self._success_steps += 1
        else:
            self._success_steps = 0

        success = self._success_steps >= self.SUCCESS_HOLD_STEPS
        if success:
            reward += 10.0

        # desired_speed 보상: 목표 EE 속도 추종 (목표 근처에서 감쇠)
        # obs layout: qpos(6)+qvel(6)+ee_pos(3)+ee_quat(4)+ee_vel_lin(3)+ee_vel_qdot(4)+desired_speed(1)
        if self.speed_weight > 0.0:
            ee_lin_vel = obs["observation"][19:22]  # ee 선속도 (Jacobian 기반)
            ee_speed   = float(np.linalg.norm(ee_lin_vel))
            gate = float(np.clip(dist / (self.success_threshold * SPEED_GATE_K), 0.0, 1.0))
            reward -= self.speed_weight * gate * (ee_speed - self._meta_speed) ** 2

        # 장애물 충돌 체크
        terminated = False
        if self._check_collision(ee_pos):
            reward    -= COLLISION_PENALTY
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

        # 타겟 설정 (FK 기반 도달 가능한 위치+자세 동시 샘플링)
        target_pos, target_quat = self._sample_target_pose()
        self._set_target(target_pos, target_quat)

        # 장애물 초기화
        self._reset_obstacles()

        # desired_speed: [0.0, 1.0] 균등 샘플 (사용자 입력 범위, observation에 포함)
        self._desired_speed = float(np.random.uniform(0.0, 1.0))

        # meta_speed: [0.1, 0.9] — 극단값 회피, 속도 보상 및 에피소드 길이 계산에 사용
        self._meta_speed = SPEED_META_MIN + self._desired_speed * (SPEED_META_MAX - SPEED_META_MIN)

        # max_episode_steps: meta_speed(=목표 EE 속도) 기반 재계산
        # = 2 × max_dist / (meta_speed × CONTROL_DT) = 100 / meta_speed
        self.max_episode_steps = int(np.clip(
            100.0 / self._meta_speed, SPEED_STEPS_MIN, SPEED_STEPS_MAX
        ))

        mujoco.mj_forward(self.model, self.data)
        self._step_count    = 0
        self._success_steps = 0

        obs = self._get_obs()
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

        # 장애물 이동
        self._step_obstacles()

        obs = self._get_obs()
        reward, success, collision_terminated, dist, angle_err = self._step_reward(obs)

        self._step_count += 1
        terminated = success or collision_terminated
        truncated  = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {
            "distance":    dist,
            "angle_error": angle_err,
            "success":     success,
            "collision":   collision_terminated,
            "step":        self._step_count,
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
