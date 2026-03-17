# M1013 RL Example

두산 로보틱스 M1013 6축 로봇 팔을 MuJoCo로 시뮬레이션하고, PPO로 Reach 태스크를 학습하는 예제입니다.

## 태스크

엔드이펙터(TCP)를 3D 공간의 랜덤 타겟 위치로 이동시키는 **Reach Task**를 강화학습으로 학습합니다.

## 환경

- **시뮬레이터**: MuJoCo 3.x
- **로봇**: Doosan Robotics M1013 (6-DOF, max reach 1.3m)
- **알고리즘**: PPO (stable-baselines3)
- **Python**: 3.12

| 항목 | 내용 |
|------|------|
| Observation | sin/cos(관절각) × 6, 관절속도 × 6, EE 위치, EE→타겟 벡터, 거리 (총 25차원) |
| Action | 각 관절 normalized torque [-1, 1] (6차원) |
| Reward | `-distance` + 근접 보너스 + 성공 시 `+10` |
| 성공 기준 | 타겟까지 5cm 이내 |

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
# MuJoCo 기본 뷰어
python -m mujoco.viewer --mjcf=m1013.xml

# 관절 모션 확인
python view.py
```

### 학습

```bash
python train.py --steps 1000000 --envs 8
```

학습 로그는 `logs/`, 모델 체크포인트는 `models/`에 저장됩니다.

```bash
# TensorBoard로 학습 모니터링
tensorboard --logdir logs
```

### 평가

```bash
# 텍스트 결과
python evaluate.py --model models/best_model --episodes 20

# MuJoCo 뷰어로 시각화
python evaluate.py --model models/best_model --render
```

## 파일 구조

```
rl-example/
├── m1013.xml       # Doosan M1013 MuJoCo 모델
├── robot_env.py    # Gymnasium 환경 (M1013Reach-v0)
├── train.py        # PPO 학습 스크립트
├── evaluate.py     # 평가 스크립트
├── view.py         # 뷰어 예제 (사인파 모션)
├── meshes/         # M1013 STL 메시 파일
│   ├── m1013_stl/       # 비주얼 메시
│   └── m1013_collision/ # 충돌 메시
├── models/         # 학습된 모델 저장
└── logs/           # TensorBoard 로그
```

## 참고

- [Doosan Robotics doosan-robot2](https://github.com/DoosanRobotics/doosan-robot2)
- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
