# M1013 SAC+HER 학습 레슨런 (Lessons Learned)

**환경**: Doosan M1013 6-DOF 로봇 팔 Reach Task (위치 + 자세 목표)
**최종 알고리즘**: SAC + HER (Torque 제어 전용)
**현재 도달 단계**: pos 6.29cm / ori 9.4° / 장애물 2개 (v5 fresh start 진행 중)

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
Torque가 Position보다 일관되게 빠름. Position의 Δθ action은 탐색 폭 제한적,
Torque는 더 자유로운 궤적 탐색 가능.

---

## Phase 2 — PPO + FK 기반 전체 Workspace 커리큘럼 (부분 성공)

### 커리큘럼 재설계 (8116c47)

구 설계의 근본 문제(EE 반경 + joint 랜덤화 결합)를 해결:

| 항목 | 구 설계 | 새 설계 |
|------|---------|---------|
| 타겟 위치 | EE 중심 반경 내 임의 점 | FK 기반 전체 workspace 임의 위치 |
| 커리큘럼 변수 | `level` → `max_r` | `success_threshold` (30cm → 0.1mm) |
| level-up 조건 | 성공률 ≥40% | 성공률 ≥70% → threshold ×0.8 |
| level-down 조건 | 성공률 <10% | 성공률 <20% → threshold ×1.2 |
| 시작 자세 | fixed zero | 완전 임의 (전체 JOINT_RANGES) |

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
    "observation":   27D,  # qpos(6) + qvel(6) + ee_pos(3) + ee_quat(4) + ee_vel(7) + desired_speed(1)
    "achieved_goal":  7D,  # ee_pos(3) + ee_quat(4)
    "desired_goal":   7D,  # target_pos(3) + target_quat(4)
    "obstacles":     70D,  # 10슬롯 × [cx,cy,cz,p1,p2,p3,type_flag]
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
# step에서는 여기에 성공 보너스 +10.0, 충돌 패널티 -5.0만 추가
```

### 3.2 n_sampled_goal 실험

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
        self.min_log_ent_coef = math.log(min_ent_coef)

    def _on_step(self):
        self.model.log_ent_coef.data.clamp_(min=self.min_log_ent_coef)
        return True
```

SAC는 `log_ent_coef`를 Adam으로 업데이트하므로 log 공간에서 클램핑하면 됨.
`ent_coef_loss`가 크게 음수(-8 수준)로 유지되지만, 바닥(0.03)이 유지되어 탐색 압력 보존.

CurriculumCallback이 target_entropy를 자동 조정하면 ent_coef 붕괴를 가속하는 것도 확인.
→ target_entropy 자동 조정 로직 전체 제거, `-1.0` 고정.

---

## Phase 5 — Init Range Curriculum

### 설계

- 초기: `init_range = ±0.5 rad (±29°)`
- 성공률 ≥70%: `init_range × 1.5` (최대 ±π = ±180°)
- ±180° 도달 후 고정

### 핵심 교훈: ±180° init_range에서 resume 금물

±180°까지 확장된 상태에서 resume하면 반복적으로 실패.
학습된 모델이 ±180° 초기화를 처리하지 못해 성공률 0%→커리큘럼 역행 반복.

**실패 패턴**:
```
resume → 성공률 0% → curriculum threshold 역행 (30cm까지) → 수천 에피소드 낭비
→ 갑자기 회복되거나, 계속 실패
```

**해결**: 문제가 지속되면 fresh start 선택. fresh start 시 ±29°에서 다시 시작하므로
커리큘럼이 빠르게 재진행됨.

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

### 6.3 SB3 LR Schedule 버그

`model.learning_rate`만 바꾸면 SB3 내부 `_update_learning_rate()`가 `lr_schedule`로 덮어씀.

```python
from stable_baselines3.common.utils import get_schedule_fn
model.learning_rate = finetune_lr
model.lr_schedule = get_schedule_fn(finetune_lr)
for opt in [model.policy.actor.optimizer,
            model.policy.critic.optimizer,
            model.ent_coef_optimizer]:
    for pg in opt.param_groups:
        pg["lr"] = finetune_lr
```

---

## Phase 7 — Torque 전용 전환 + 동적 장애물

### Position 제거 결정

Position 제어(Δθ action)는 Torque 대비 일관되게 느리고, 동적 장애물 추가 시 복잡도가 높아짐.
→ Torque 전용으로 정리, Position 관련 파일 `legacy/` 이동 후 최종 삭제.

### 동적 장애물 설계

- 100개 형상 풀 (구 34 + 박스 33 + 캡슐 33)
- 에피소드마다 비복원 추출로 최대 10개 활성화
- XML에 슬롯당 3개 geom(sphere/box/capsule) 미리 할당, rgba=0으로 숨김
- 활성 타입만 rgba 표시, 런타임에 `model.geom_size`로 크기 조정

### 충돌 체크 확장: EE만 → 전체 링크

초기 EE만 체크 → 로봇 링크가 장애물에 닿아도 충돌 미감지.

**해결**: 12개 구형 근사 포인트:
- 7개 링크 바디 (base_link~link6) + EE
- 긴 세그먼트(link2→3, link3→4) 중간점 각 2개

### ObstacleAwareExtractor

장애물 정보를 별도 MLP로 처리해 로봇 상태와 concat:
```
robot_obs(41D) → Linear(256) → ReLU → Linear(256) → ReLU → 256D
obstacles(70D) → Linear(128) → ReLU → Linear(64)  → ReLU →  64D
                                                   concat → 320D
```
단순 concat 대비 장애물 특징을 독립적으로 추출해 학습 효율 향상.

---

## Phase 8 — desired_speed 보상 설계 시행착오

### 초기 시도: 즉시 적용 (실패)

`speed_weight=0.2`를 best_model resume 시 즉시 적용 → ep_rew_mean -393, 커리큘럼 역행.

**원인**: 기존 모델이 speed 추종 없이 학습됐는데 갑자기 강한 속도 패널티 추가 → 보상 landscape 급변.

### 설계 원칙: desired_speed도 커리큘럼으로

```
speed_weight = 0.0 (초기)
pos_threshold < 5cm 달성 후 성공률 ≥70%마다 +0.05 (최대 0.2)
```

위치 먼저 충분히 학습한 뒤 속도 추종 점진 도입.

### meta_speed 설계

```python
meta_speed = SPEED_META_MIN + desired_speed × (SPEED_META_MAX - SPEED_META_MIN)
           = 0.1 + desired_speed × 0.8   # [0.1, 0.9] m/s
```

`gate = clip(dist / (threshold × 2), 0, 1)` — 목표 근처에서 속도 보상 감쇠.
목표에 가까울수록 속도보다 정확도에 집중하도록 설계.

### max_episode_steps 동적 조정

```python
max_episode_steps = int(clip(100.0 / meta_speed, 150, 500))
```

느린 desired_speed → 더 많은 스텝 허용.

---

## Phase 9 — 타겟 자세 도달 불가 문제

### 문제 발견

`evaluate.py`로 시각화 시 로봇이 타겟을 향해 가는데 절대 도달 못하는 경우가 발생.

**원인**: `_sample_target()`은 FK 기반으로 **위치**만 실제 도달 가능하게 샘플링하지만,
`_sample_target_quat()`은 **완전히 랜덤한 quaternion**을 반환.

특정 위치에서 가능한 EE 자세는 로봇의 기구학에 의해 제한되는데, 임의 자세를 목표로 설정하면
물리적으로 불가능한 자세가 목표가 됨.

### 해결

두 함수를 `_sample_target_pose()`로 통합:

```python
def _sample_target_pose(self):
    # FK로 관절 설정을 랜덤 샘플링
    for _ in range(50):
        random_qpos → mj_forward → candidate_pos, candidate_quat
        if dist(candidate_pos, current_pos) <= max_dist:
            return candidate_pos, candidate_quat  # 같은 FK에서 추출
    return current_pos, current_quat  # fallback
```

같은 관절 설정에서 위치와 자세를 동시에 가져오므로 도달 가능성 100% 보장.

---

## Phase 10 — 학습 전략 최적화

### 2M → 100M step 전환

2M 스텝마다 재시작하는 방식의 문제:
- 매 재시작마다 `warmup_episodes=2000` 낭비
- `learning_starts=30,000` 반복
- 모델이 새 환경에 재적응하는 시간 낭비

→ 한 번에 100M 스텝으로 시작. 재시작 필요 시에도 100M으로 이어서 진행.

### Fresh Start vs Resume 선택 기준

| 상황 | 선택 |
|------|------|
| 환경 구조 변경 (obs 차원, 보상 함수 대폭 변경) | **Fresh Start** |
| 하이퍼파라미터 조정, 버그 수정 | Resume |
| init_range ±180° 상태에서 성공률 0% 지속 | **Fresh Start** |
| 커리큘럼 상태 양호 + 모델 성능 확인됨 | Resume |

### evaluate.py --obstacles 고정 동작

`--obstacles N` 미지정 시 `randint(0, max_obs_count+1)`로 0~N 중 랜덤 선택.
`--obstacles 1` 지정 시에도 0개가 나오는 문제 발견.

**해결**: `fixed_obs_count` 속성 추가:
```python
if self.fixed_obs_count is not None:
    n_active = int(clip(self.fixed_obs_count, 0, MAX_OBSTACLES))
else:
    n_active = randint(0, self.max_obs_count + 1)
```

---

## 커리큘럼 진행 요약 (v5 fresh start 기준)

| 단계 | pos threshold | ori threshold | init_range | obs | 비고 |
|------|--------------|---------------|-----------|-----|------|
| 1 | 30.0cm | 45.0° | ±29° | 0 | 초기 |
| 2 | 24.0cm | 36.0° | ±43° | 0 | |
| 3 | 19.2cm | 28.8° | ±64° | 0 | |
| 4 | 15.4cm | 23.0° | ±97° | 0 | |
| 5 | 12.3cm | 18.4° | ±145° | 0 | |
| 6 | 9.83cm | 14.7° | ±180° | 0 | init_range 최대 도달 |
| 7 | 7.86cm | 11.8° | ±180° | 1 | **장애물 첫 등장** |
| 8 (현재) | 6.29cm | 9.4° | ±180° | 2 | 진행 중 |

fresh start 후 약 2.8M 스텝(~28,000 에피소드)에서 장애물 2개, pos 6.29cm 달성.

---

## 현재 모델 상태

| 항목 | 값 |
|------|-----|
| 알고리즘 | SAC + HER (Torque) |
| pos threshold | 6.29cm |
| ori threshold | 9.4° |
| init_range | ±180° |
| 장애물 | 2/10 |
| speed_weight | 0.00 (pos < 5cm 미달성) |
| ent_coef | 0.03 (floor 유지) |
| ep_rew_mean | ~-18 |

---

## 핵심 교훈 요약

1. **커리큘럼 변수와 목표를 분리하지 마라**: 타겟 분포와 성공 기준이 별도로 움직이면 학습이 어느 쪽을 향하는지 알기 어려움. `success_threshold` 하나로 통합이 효과적.

2. **FK를 쓸 거면 위치와 자세를 함께**: FK 위치만 맞추고 자세는 랜덤으로 하면 도달 불가 목표가 생성됨.

3. **보상 변경은 점진적으로**: 기존 모델에 새 보상항 즉시 추가 시 학습 붕괴. 커리큘럼으로 단계적 도입.

4. **ent_coef 자동 조절 믿지 마라**: SAC의 자동 엔트로피 조절은 탐색이 충분한 경우에도 ent_coef를 0에 가깝게 낮춤. 최솟값 클램핑 필수.

5. **Resume 시 warmup 필수**: replay buffer 재충전 기간 동안 curriculum이 역행하는 것을 막아야 함.

6. **±180° init_range에서 resume = 높은 실패 확률**: 커리큘럼 상태가 좋지 않으면 fresh start가 더 빠름.

7. **PPO보다 SAC+HER이 이 태스크에 압도적**: HER의 goal resampling이 희소 보상 문제를 효과적으로 해결.
