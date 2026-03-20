# M1013 SAC+HER 학습 레슨런 (Lessons Learned)

**작성일**: 2026-03-21
**환경**: Doosan M1013 6-DOF 로봇 팔 Reach Task (위치 + 자세 목표)
**최종 알고리즘**: SAC + HER
**현재 도달 단계**: Position 22.1cm/33.2°, Torque 7.5cm/11.3°

---

## Phase 1 — PPO + EE-반경 커리큘럼 (실패)

### 초기 설계 (ff52d03)

- **알고리즘**: PPO, `MlpPolicy`, n_envs=8
- **Observation 25D**: joint sin/cos(12) + qvel(6) + ee_pos(3) + ee→target(3) + dist(1)
- **Action**: normalized torque [-1, 1] × 6
- **커리큘럼**: EE 중심 반경 `max_r = 8cm + 252cm × level`, level ±0.01 조정
- **목표**: 위치만 (자세 없음)

### 왜 실패했는가

**구조적 천장**: level 0.01 진입 시마다 즉각 붕괴.

- Joint 랜덤화 범위 = `±level × JOINT_RANGES`
  → level 0.01 → ±0.063 rad → EE 불확실성 수십 cm
- EE 중심 타겟 반경(8~10cm)이 joint 노이즈로 인한 EE 불확실성보다 작음
  → 타겟이 실질적으로 "이미 EE 밖"에 있는 상태로 시작 → 성공률 60%→10% 급락

**핵심 모순**: level이 올라갈수록 joint 랜덤화도 강해져 어려움의 종류가 바뀌는 구조.
50M 스텝, 하루 이상 학습해도 실질적으로 10cm 반경 문제만 반복.

### Phase 1 실험에서 얻은 교훈

| 실험 | 결과 |
|------|------|
| LR 선형 감소 스케줄 `3e-4×(1-progress)` | level-up 후 LR 부족 → 적응 불가. 상수 `3e-4`로 변경 |
| `JOINT_RAND_MIN` 노이즈 추가 | 초기 수렴 방해. position은 초기 단계에서 EE 불확실성에 매우 민감 |
| level-up 임계값 40%→50% | 달성 자체가 드물어 오히려 악화 |
| level-up 스텝 +0.05 | 한 번에 max_r이 2.6배 급증 → 즉각 붕괴 |
| Value function 과특화 | `explained_variance≈0.98`이 되면 policy gradient 신호 약해져 정체 |

**Position vs Torque 비대칭**:
Torque가 Position보다 일관되게 빠름. Position의 Δθ action은 탐색 폭 제한적, Torque는 더 자유로운 궤적 탐색 가능.

---

## Phase 2 — PPO + FK-기반 전체 Workspace 커리큘럼 (부분 성공)

### 커리큘럼 재설계 (8116c47)

구 설계의 근본 문제(EE 반경 + joint 랜덤화 결합)를 해결:

| 항목 | 구 설계 | 새 설계 |
|------|---------|---------|
| 타겟 위치 | EE 중심 반경 내 임의 점 | FK 기반 전체 workspace 임의 위치 |
| 커리큘럼 변수 | `level` → `max_r` | `success_threshold` (30cm → 0.1mm) |
| level-up 조건 | 성공률 ≥40% | 성공률 ≥70% → threshold ×0.8 |
| level-down 조건 | 성공률 <10% | 성공률 <20% → threshold ×1.2 |
| 시작 자세 | fixed zero | 완전 임의 (전체 JOINT_RANGES) |

**장점**: 처음부터 전체 workspace 학습, threshold가 실제 목표 정밀도와 직결.

### 추가된 보상 요소 (91a221b)

- proximity bonus: `exp(-dist/0.3)`
- success hold requirement (커리큘럼 연동 0.1s→5s)
- joint limit penalty
- 자기충돌 버그 수정: joint2가 수평 이하로 못 내려가는 문제 → 모든 collision geom 비활성화

### 한계

PPO는 replay buffer가 없어 sample efficiency가 낮음. 희소한 성공 신호(특히 tight한 threshold에서)를 충분히 학습하기 어려움.
→ **SAC + HER로 전환 결정**

---

## Phase 3 — SAC + HER 도입 (구조 재설계)

### 알고리즘 전환 이유

- HER(Hindsight Experience Replay): 실패한 trajectory도 "그 위치가 목표였다면 성공"으로 재라벨링 → sample efficiency 대폭 향상
- SAC: off-policy, entropy 정규화로 탐색/활용 균형 자동 조절

### HER을 위한 환경 재설계

**GoalEnv 구조로 변경**:
```python
observation = {
    "observation":   26D,  # qpos(6) + qvel(6) + ee_pose(7) + ee_vel(7)
    "achieved_goal":  7D,  # ee_pos(3) + ee_quat(4)
    "desired_goal":   7D,  # target_pos(3) + target_quat(4)
}
```

**자세(orientation) 목표 추가**: 위치만 맞추는 것에서 위치+자세 동시 달성으로 확장.

### 3.1 HER 보상 불일치 문제 — 가장 치명적인 초기 버그

**문제**: step 내부 `_compute_reward()`와 HER용 `compute_reward()`가 서로 다른 로직.
- step 보상: 9개 항목 (거리, 각도, 진행도, 정지 패널티, 성공 보너스 등)
- HER 보상: 2개 항목만 (거리 + 각도)

HER은 과거 transition에 새 목표 적용 후 `compute_reward()`로 재계산 → 두 함수가 다르면 학습 신호 불일치.

**해결**: 단일 함수로 통합.
```python
def compute_reward(self, achieved_goal, desired_goal, info):
    pos_dist   = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
    angle_diff = quaternion_angle(achieved_goal[3:], desired_goal[3:])
    return -(pos_dist + 0.3 * angle_diff)
# step에서는 여기에 성공 보너스 +10.0만 추가
```

### 3.2 Observation — EE 중심 설계

관절 공간 위주 → EE 중심 26D:
```
qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7) = 26D
```
- `ee_vel = lin_vel(3) + qdot(4)`: Jacobian 기반 선속도 + 자세변화율
- goal과 동일한 공간(Cartesian)에서 표현 → HER 학습 효율 향상

### 3.3 커리큘럼 목표 샘플링 — 방향 오류 수정

**문제**: `max_dist = (0.30 / threshold) × 0.60` → threshold 감소할수록 목표가 더 멀어지는 역방향.
**해결**: `max_dist = min(1.0, threshold × 10)` — threshold에 비례해 목표 난이도 함께 감소.

### 3.4 n_sampled_goal 실험

| 설정 | 결과 |
|------|------|
| `n_sampled_goal=4` | 정상 학습 |
| `n_sampled_goal=8` | position 8.2%→3.8%, torque 19.6%→2.2% 급락 |

goal이 너무 많으면 critic 불안정. `n_sampled_goal=4` 유지.

---

## Phase 4 — ent_coef 붕괴 문제

### 현상

SAC 자동 조절로 `ent_coef`가 0.006까지 하락 → 사실상 결정론적 정책 → 탐색 없음 → 로컬 미니멈 고착.

```
ent_coef = 0.006, H(π) ≈ 1 nat
→ entropy 기여 = 0.006  vs  보상 = -20 ~ -100
→ entropy 항이 무시할 수준
```

### 시도한 해결책

| 방법 | 결과 |
|------|------|
| `target_entropy="auto"` (기본값 -6.0) | ent_coef→0.005 붕괴 |
| `target_entropy=-3.0` | ent_coef→0.006 붕괴 |
| `target_entropy=-1.0` | 콜백이 -2.6으로 재조정, 여전히 붕괴 |
| target_entropy 자동 조정 제거 | 근본 해결 안 됨 |
| **ent_coef 최솟값 0.03 클램핑** | **✅ 효과적** |

### 최종 해결: EntCoefFloorCallback

```python
class EntCoefFloorCallback(BaseCallback):
    def __init__(self, min_ent_coef=0.03):
        import math
        self.min_log_ent_coef = math.log(min_ent_coef)

    def _on_step(self):
        self.model.log_ent_coef.data.clamp_(min=self.min_log_ent_coef)
        return True
```

SAC는 `log_ent_coef`를 Adam으로 업데이트하므로 log 공간에서 클램핑하면 됨.
`ent_coef_loss`가 크게 음수(-14 수준)로 유지되지만, 바닥(0.03)이 유지되어 탐색 압력 보존.

### target_entropy 자동 조정 제거

CurriculumCallback이 성공률 ≥70% 시 target_entropy를 -0.2씩 낮춰 ent_coef 붕괴를 가속하는 것을 확인.
→ target_entropy 자동 조정 로직 전체 제거, `-1.0` 고정.

---

## Phase 5 — Init Range Curriculum

### 목적

임의 시작점 → 임의 목표점 이동 능력 학습 (단순 reach가 아닌 general motion).

### 설계

- 초기: `init_range = ±0.5 rad (±29°)`
- 성공률 ≥70%: `init_range × 1.5` (최대 ±π=±180°)
- 성공률 <20%: 유지 (축소 안 함 — 한번 확장된 범위는 유지)

### 현재 결과

- Position: ±97° (초기 ±29° 대비 3.3배 확장)
- Torque: ±43°

---

## Phase 6 — Resume 함정들

### 6.1 learning_starts 재설정

Resume 시 replay buffer가 비어있으므로 반드시 재설정:
```python
model.learning_starts = model.num_timesteps + 30_000
```

### 6.2 Resume 중 Curriculum 완화

learning_starts 기간 동안 랜덤 액션 → 성공률 0% → curriculum callback이 threshold를 계속 완화.

**해결**: `warmup_episodes=2000` — resume 시 처음 2000 에피소드 동안 curriculum 조정 건너뜀.

```python
if self.episodes < self.warmup_episodes:
    # 로그만 출력, curriculum 변경 없음
    return True
```

### 6.3 Threshold 저장 파일 오염

Bad run이 threshold 파일을 덮어쓴 경우 수동 복원 필요:
```bash
echo "0.192" > models/position/success_threshold.txt
python3 -c "import math; print(math.radians(28.8))" > models/position/ori_threshold.txt
```

### 6.4 SB3 LR Schedule 버그

`model.learning_rate`만 바꾸면 SB3 내부 `_update_learning_rate()`가 `lr_schedule`로 덮어씀.

```python
from stable_baselines3.common.utils import get_schedule_fn
model.learning_rate = finetune_lr
model.lr_schedule = get_schedule_fn(finetune_lr)
for opt in [model.policy.actor.optimizer,
            model.policy.critic.optimizer,
            model.ent_coef_optimizer]:  # model.policy.ent_coef_optimizer → AttributeError!
    for pg in opt.param_groups:
        pg["lr"] = finetune_lr
```

---

## 커리큘럼 진행 요약

### Torque (더 빠른 모델)

| 단계 | pos threshold | ori threshold | 최고 성공률 |
|------|--------------|---------------|-------------|
| 1 | 30.0cm | 45.0° | 76.4% |
| 2 | 24.0cm | 36.0° | 70.2% |
| 3 | 19.2cm | 28.8° | 70.2% |
| 4 | 15.4cm | 23.1° | 72.2% |
| 5 | 12.3cm | 18.5° | 69.2% |
| 6 | 9.8cm  | 14.7° | 75.2% |
| 7 | 7.9cm  | 11.8° | 70.2% |
| 8 (현재) | 7.5cm | 11.3° | 43%→ 상승 중 |

### Position

| 단계 | pos threshold | ori threshold | 최고 성공률 |
|------|--------------|---------------|-------------|
| 1 | 30.0cm | 45.0° | 77.8% |
| 2 | 24.0cm | 36.0° | 76.4% |
| 3 | 22.1cm (현재) | 33.2° | 70.6% → 51.4% 적응 중 |

---

## 현재 모델 상태 (2026-03-21 기준)

| | Position | Torque |
|--|----------|--------|
| threshold | 22.1cm / 33.2° | 7.5cm / 11.3° |
| init_range | ±97° | ±43° |
| ent_coef | 0.03 (클램핑) | 0.03 (클램핑) |
| best_model | `models/position/best_model.zip` | `models/torque/best_model.zip` |

---

## 남은 과제

1. **Position이 Torque보다 뒤처짐**: Δθ action 방식이 탐색 폭을 제한하는 것으로 추정. 추가 분석 필요.

2. **ent_coef_loss 대형 음수 (-14)**: SAC가 계속 ent_coef를 낮추려 하는 상태 — tight한 threshold에서 수렴이 완료된 후 floor를 낮출 것인지 검토 필요.

3. **Init range 완전 일반화**: ±180°까지 도달해야 임의 시작→끝 이동 가능.

4. **최종 평가**: `python evaluate.py --render`로 실제 달성 정밀도 검증 필요.
