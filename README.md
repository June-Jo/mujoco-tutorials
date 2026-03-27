# M1013 RL Example

두산 로보틱스 M1013 6축 로봇 팔을 MuJoCo로 시뮬레이션하고, **SAC + HER**로 Reach 태스크를 학습하는 예제입니다.

> **Torque 제어 전용.**

## 태스크

엔드이펙터(TCP)를 3D 공간의 랜덤 타겟 **위치 + 자세**로 이동시키는 Reach Task를 강화학습으로 학습합니다.

- 시작 자세: 6개 관절 임의 초기화 (커리큘럼에 따라 ±29°→±180° 확장)
- 타겟: **FK 기반 도달 가능한** 위치 + 자세 동시 샘플링 (같은 관절 설정에서 EE pose 추출 → 항상 reachable 보장)
- 성공 기준: 위치 오차 < threshold AND 자세 오차 < ori_threshold (커리큘럼 기반 자동 조정)
- 동적 장애물 회피: 에피소드마다 0~max_obs_count개 장애물이 workspace를 이동

## 환경

- **시뮬레이터**: MuJoCo 3.x (500Hz 물리, 10 sub-steps → 50Hz 제어)
- **로봇**: Doosan Robotics M1013 (6-DOF, max reach 1.3m)
- **알고리즘**: SAC + HER (Hindsight Experience Replay), stable-baselines3
- **Python**: 3.12

### Observation Space

GoalEnv Dict 구조:

| 키 | 차원 | 내용 |
|---|---|---|
| `observation` | 26D | `qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7)` |
| `achieved_goal` | 7D | `ee_pos(3) + ee_quat(4)` |
| `desired_goal` | 7D | `target_pos(3) + target_quat(4)` |
| `obstacles` | 70D | 10슬롯 × `[cx, cy, cz, p1, p2, p3, type_flag]` |

### Action Space

6개 관절 정규화 토크 `[-1, 1]` (gear 비율 기반 실제 토크 변환)

### Reward

```
reward = -(pos_dist + 1.0 × angle_error)  # 매 스텝
       + 10.0                             # 성공 시
       - 5.0                              # 충돌 시 (에피소드 종료)
```

## 동적 장애물

100개 형상 풀(구 34 + 박스 33 + 캡슐 33)에서 에피소드마다 최대 10개를 비복원 추출해 배치합니다.

- 장애물은 매 스텝 이동하며, workspace 경계에서 반사
- 충돌 체크: 로봇 링크 전체(base_link~link6 + EE, 긴 세그먼트 중간점 포함) → 12개 구형 근사 포인트
- 커리큘럼: 위치 threshold < 10cm 달성 후 장애물 수를 0→10으로 단계적 증가

## 네트워크 구조

```
ObstacleAwareExtractor
  robot_net:    (26+7+7=40)D → Linear(512) → ReLU → Linear(512) → ReLU  → 512D
  obstacle_net: 70D          → Linear(256) → ReLU → Linear(128) → ReLU  → 128D
                                                              concat → 640D
SAC Actor/Critic: 640D → [512, 512, 512]
```

## 커리큘럼

성공률 기준으로 자동 조정합니다.

| 조건 | 동작 |
|---|---|
| 성공률 ≥ 85% | pos / ori / obs / init_range 중 **하나를 랜덤 선택**해 난이도 상승 |
| 성공률 < 20% | pos/ori threshold × 1.2 |

각 축의 전진 규칙:
- `pos`: success_threshold × 0.8
- `ori`: ori_threshold × 0.8
- `obs`: max_obs_count + 1 (최대치면 후보 제외)
- `init_range`: init_range × 1.5 (최대치면 후보 제외)

| 변수 | 초기값 | 범위 |
|---|---|---|
| pos_threshold | 30 cm | → 0.1 mm |
| ori_threshold | 45° | → 0.1° |
| init_range | ±29° | → ±180° |
| max_obs_count | 0 | → 10 |

커리큘럼 상태는 `models/torque/*.txt`에 저장되어 resume 시 자동 복원됩니다.

## 설치

```bash
git clone <repo-url>
cd rl-example
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 사용법

### 학습

```bash
# 처음부터
python train.py --steps 100000000 --envs 16

# 이어서 학습 (best_model 기준)
python train.py --steps 100000000 --envs 16 --resume models/torque/best_model
```

학습 로그는 `logs/`, 모델 체크포인트는 `models/torque/`에 저장됩니다.

```bash
# TensorBoard 모니터링
tensorboard --logdir logs
```

### 평가 및 시각화

```bash
# 텍스트 평가 (저장된 curriculum 상태 자동 복원)
python evaluate.py

# MuJoCo 뷰어로 시각화
python evaluate.py --render

# 장애물 수 고정 지정 (정확히 N개, 랜덤 아님)
python evaluate.py --obstacles 5

# 에피소드 수 / 재생 속도 지정
python evaluate.py --render --episodes 10 --speed 0.5
```

## 파일 구조

```
rl-example/
├── m1013.xml              # MuJoCo 씬 (로봇 + 10슬롯×3geom 장애물 + target mocap)
├── robot_env.py           # Gymnasium GoalEnv (M1013Reach-v0)
├── train.py               # SAC+HER 학습 스크립트 (ObstacleAwareExtractor 포함)
├── evaluate.py            # 평가 및 시각화 스크립트
├── LESSONS_LEARNED.md     # 학습 과정 회고록
├── meshes/                # M1013 STL/DAE 메시 파일
├── models/torque/         # 학습된 모델 및 커리큘럼 상태
│   ├── best_model.zip
│   ├── vecnormalize.pkl
│   ├── success_threshold.txt
│   ├── ori_threshold.txt
│   ├── init_range.txt
│   └── max_obs_count.txt
└── logs/                  # TensorBoard 로그
```

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `learning_rate` | 3e-4 | 학습률 |
| `buffer_size` | 1,000,000 | Replay buffer 크기 |
| `batch_size` | 512 | 미니배치 크기 |
| `tau` | 0.005 | Target network soft update 비율 |
| `gamma` | 0.95 | 할인율 |
| `target_entropy` | -3.0 | SAC 목표 엔트로피 (결정론적 정책 유도) |
| `ent_coef` 초기값 | `auto_0.2` | 엔트로피 계수 초기값 |
| `ent_coef` floor (평상시) | 없음 | SAC optimizer가 자유롭게 낮출 수 있음 |
| `ent_coef` floor (부스트) | 0.15 | 정체 감지 시 일시 상향 (2,000 에피소드) |
| `action_noise` | N(0, 0.1) | 롤아웃 탐색 노이즈 |
| `n_sampled_goal` | 4 | HER 힌트 goal 수 |
| `goal_selection_strategy` | future | HER 목표 선택 전략 |
| `net_arch` | [512, 512, 512] | Actor/Critic 네트워크 구조 |
| `max_episode_steps` | 200 | 에피소드 최대 스텝 수 |
| `n_envs` | 16 | 병렬 환경 수 |

### 적응형 엔트로피 제어

`EntCoefFloorCallback`이 ent_coef 하한을 동적으로 조절합니다.

- **평상시 floor 없음**: SAC optimizer가 `target_entropy=-3.0`을 향해 ent_coef를 자유롭게 낮춤 — 정밀 제어 구간에서 결정론적 exploit 허용
- **부스트 (2,000 에피소드)** `boost_floor=0.15`: 정체 감지 시 자동 발동
  - 정체 감지 (`stagnation_windows=10`, 성공률 10 윈도우 연속 개선 없음) — 로컬 미니멈 탈출

## 참고

- [Doosan Robotics doosan-robot2](https://github.com/DoosanRobotics/doosan-robot2)
- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
- [HER Paper (Andrychowicz et al., 2017)](https://arxiv.org/abs/1707.01495)
