# M1013 RL Example

두산 로보틱스 M1013 6축 로봇 팔을 MuJoCo로 시뮬레이션하고, PPO로 Reach 태스크를 학습하는 예제입니다.

## 태스크

엔드이펙터(TCP)를 3D 공간의 랜덤 타겟 위치로 이동시키는 **Reach Task**를 강화학습으로 학습합니다.

- 시작 자세: 6개 관절 전체 범위 내 완전 임의
- 타겟 위치: FK 기반 전체 workspace 임의 샘플링 (바닥 아래 포함)

## 환경

- **시뮬레이터**: MuJoCo 3.x
- **로봇**: Doosan Robotics M1013 (6-DOF, max reach 1.3m)
- **알고리즘**: PPO (stable-baselines3)
- **Python**: 3.12

| 항목 | 내용 |
|------|------|
| Observation | 관절각 × 6, 관절속도 × 6, EE 위치 × 3, EE→타겟 벡터 × 3, 거리 × 1 (총 19차원) |
| Action (position) | 각 관절 Δθ [-1, 1] → [-0.05, 0.05] rad (6차원) |
| Action (torque) | 각 관절 정규화 토크 [-1, 1] (6차원) |
| Reward | `(prev_dist - dist) × 10` + 성공 보너스 `+10` - 생존 페널티 `0.01` - 액션 smoothness |
| 성공 기준 | Curriculum 기반 (초기 30cm, 최소 0.1mm) |

## 커리큘럼

성공률에 따라 success threshold를 자동 조정합니다.

| 조건 | 동작 |
|------|------|
| 성공률 ≥ 70% (500 에피소드 기준) | threshold × 0.5 (더 어렵게) |
| 성공률 < 20% | threshold × 2.0 (더 쉽게, 최대 30cm) |
| 범위 | 30cm → 0.1mm |

threshold는 `models/{mode}/success_threshold.txt`에 저장되어 resume 시 복원됩니다.

## 설치

```bash
git clone <repo-url>
cd rl-example
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 사용법

### 로봇 모델 확인

```bash
# 관절 사인파 모션 확인
python view.py              # position 제어
python view.py --torque     # torque 제어
```

### 학습

```bash
# position 제어 (fresh start)
python train.py --steps 2000000 --envs 8

# torque 제어 (fresh start)
python train.py --steps 2000000 --envs 8 --torque

# 이어서 학습 (best_model 기준)
python train.py --steps 2000000 --envs 8 --resume models/position/best_model
python train.py --steps 2000000 --envs 8 --torque --resume models/torque/best_model
```

학습 로그는 `logs/`, 모델 체크포인트는 `models/`에 저장됩니다.

```bash
# TensorBoard로 학습 모니터링
tensorboard --logdir logs
```

### 평가 및 시각화

```bash
# position best_model 평가 (텍스트)
python evaluate.py

# torque best_model 평가
python evaluate.py --torque

# MuJoCo 뷰어로 시각화
python evaluate.py --render
python evaluate.py --render --torque

# 에피소드 수 지정
python evaluate.py --render --episodes 10
```

## 파일 구조

```
rl-example/
├── m1013_position.xml  # Doosan M1013 MuJoCo 모델 (position 제어)
├── m1013_torque.xml    # Doosan M1013 MuJoCo 모델 (torque 제어)
├── robot_env.py        # Gymnasium 환경 (M1013Reach-v0, M1013Reach-Torque-v0)
├── train.py            # PPO 학습 스크립트
├── evaluate.py         # 평가 및 시각화 스크립트
├── view.py             # 뷰어 예제 (사인파 모션)
├── meshes/             # M1013 STL 메시 파일
│   ├── m1013_stl/           # 비주얼 메시
│   └── m1013_collision/     # 충돌 메시
├── models/             # 학습된 모델 저장
│   ├── position/
│   │   ├── best_model.zip
│   │   └── success_threshold.txt
│   └── torque/
│       ├── best_model.zip
│       └── success_threshold.txt
├── logs/               # TensorBoard 로그
└── reports/            # 학습 이력 및 분석
    ├── position.md
    ├── torque.md
    └── retrospective.md
```

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `n_steps` | 2048 | rollout 길이 |
| `batch_size` | 256 | 미니배치 크기 |
| `n_epochs` | 10 | PPO 업데이트 반복 횟수 |
| `learning_rate` | 3e-4 | 학습률 (고정) |
| `ent_coef` | 0.02 | 엔트로피 계수 (탐색 장려) |
| `gamma` | 0.99 | 할인율 |
| `gae_lambda` | 0.95 | GAE λ |
| `net_arch` | [256, 256, 128] | policy/value 네트워크 구조 |
| `log_std_init` | 0.0 | 초기 액션 표준편차 (std=1.0) |
| `MAX_DELTA` | 0.05 rad | 스텝당 최대 관절 이동량 (position) |
| `max_episode_steps` | 500 | 에피소드 최대 길이 |

## 참고

- [Doosan Robotics doosan-robot2](https://github.com/DoosanRobotics/doosan-robot2)
- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
