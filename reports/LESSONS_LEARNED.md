# M1013 SAC+HER 학습 레슨런 (Lessons Learned)

**환경**: Doosan M1013 6-DOF 로봇 팔 Reach Task (위치 + 자세 목표)
**최종 알고리즘**: SAC + HER (Torque 제어 전용)
**현재 도달 단계**: pos 9.83cm / ori 14.7° / 장애물 0/10 (v6 fresh start ~16M 스텝 진행 중)

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

## Phase 11 — Resume 붕괴 사건 및 v5 Fresh Start

### Resume 붕괴 (v5 이전)

9.83cm / obs 2 상태의 best_model을 `--resume`으로 이어서 학습 시도.

**붕괴 원인**:
```
resume → learning_starts = num_timesteps + 30,000 = 6.74M
→ 약 6M 스텝 동안 랜덤 액션 실행
→ 성공률 0% → curriculum callback이 threshold를 30cm까지 역행
→ ep_rew_mean -330, ep_len 180으로 완전 붕괴
```

학습이 재개된 것처럼 보이지만 6M 스텝 동안 정책이 전혀 업데이트되지 않고 쓰레기 데이터가 replay buffer에 가득 찬 상태.

**해결**: 프로세스 강제 종료 후 커리큘럼 파일 전부 초기값으로 리셋, `--resume` 없이 fresh start.

**교훈**: `learning_starts`가 `num_timesteps` 기준으로 재설정되면 대규모 모델에서는 수백만 스텝 동안 업데이트가 없음. 이 기간 동안 curriculum이 역행하면 회복이 거의 불가능.

### v5 Fresh Start 설정 변경

| 항목 | 이전 | v5 |
|------|------|-----|
| n_envs | 8 | **16** |
| advance 조건 | 90% | **85%** (obs=9 정체 경험 후 조정) |
| 학습 방식 | resume 반복 | --resume 없이 100M 스텝 단일 run |

### Advance 조건 90% → 85% 조정 이유

초기 run에서 obs=9 스테이지에서 성공률이 84~89% 구간에서 수백만 스텝 정체.
90% 조건은 달성 불가 수준이었음 → 85%로 완화.

v5에서 6.29cm 스테이지도 80~83%에서 수백만 스텝 정체했으나, 85% 조건은 결국 자연 돌파 (2.74M 스텝 소요).

### 순차 커리큘럼 동작 확인

obs 순차 증가(`obs_pending` 로직)가 예상대로 동작:
- pos threshold < 10cm 조건 충족 후 obs 0→10 순차 증가
- obs 10 완료 후에야 pos/ori threshold 감소 시작
- obs 0→10 완료까지 약 3.57M 스텝 (이후 threshold 감소가 더 어려움)

---

## 커리큘럼 진행 요약 (v5 fresh start 기준)

| 단계 | pos threshold | ori threshold | init_range | obs | 스텝(누적) | 비고 |
|------|--------------|---------------|-----------|-----|-----------|------|
| 1 | 30.0cm | 45.0° | ±29° | 0 | 0 | 초기 |
| 2 | 24.0cm | 36.0° | ±43° | 0 | ~0.3M | |
| 3 | 19.2cm | 28.8° | ±64° | 0 | ~0.6M | |
| 4 | 15.4cm | 23.0° | ±97° | 0 | ~0.9M | |
| 5 | 12.3cm | 18.4° | ±145° | 0 | ~1.2M | |
| 6 | 9.83cm | 14.7° | ±180° | 0 | ~1.5M | init_range 최대 도달 |
| 7 | 7.86cm | 11.8° | ±180° | 1~10 | ~1.5M~3.57M | obs 순차 증가 구간 |
| 8 | 6.29cm | 9.44° | ±180° | 10 | ~5.57M | obs 완료, threshold 감소 시작 |
| 9 | 5.03cm | 7.55° | ±180° | 10 | ~8.31M | |
| 10 | 4.03cm | 6.04° | ±180° | 10 | ~33M 이전 | |
| 11 (현재) | 4.03cm | 6.04° | ±180° | 10 | 33M+ | 적응 중 (63~67%) |

**장애물 순차 진행**: obs 0→10은 pos threshold가 10cm 이하인 상태에서 성공률 ≥85%마다 1개씩 증가.
obs 0→10 완료까지 약 3.57M 스텝 소요 (초기 단계 중 가장 빠른 구간).

**6.29cm 스테이지가 가장 긴 정체**: obs 10개 + 6.29cm 동시 요구 → 약 **2.74M 스텝** 소요.
성공률이 80~83% 구간에서 진동하며 85% 조건 돌파까지 오래 걸림.

---

## Phase 12 — v5 정체 돌파: v6 아키텍처 전환

### v5 정체 현상

v5 fresh start에서 9.83cm 스테이지 진입 후 약 **20M 스텝** 동안 성공률 50~57%에서 탈출하지 못함.

**분석된 원인**: 네트워크 표현력 부족 + 탐색 부족의 복합.
- `ent_coef` floor 0.03은 너무 낮아 사실상 결정론적 정책 → 로컬 미니멈 고착
- `target_entropy=-3.0` + floor 0.03 조합: SAC가 exploit 방향으로 강하게 수렴 후 재탐색 없음

### v6 아키텍처 변경

| 항목 | v5 | v6 |
|------|----|----|
| ObstacleAwareExtractor robot_out | 256 | **512** |
| ObstacleAwareExtractor obstacle_out | 64 | **128** |
| features_dim | 320 | **640** |
| net_arch | [256, 256, 256] | **[512, 512, 512]** |
| target_entropy | -3.0 | **-1.0** (더 높은 탐색 목표) |
| ent_coef 초기값 | auto_0.1 | **auto_0.2** |
| ent_coef floor | 0.03 | **0.15** |
| action_noise | 없음 | **N(0, 0.1)** 추가 |
| 커리큘럼 advance | 70%→85%→90%→85% 변천 | 85% 유지 |

**결과**: v6에서 30cm → 9.83cm 도달까지 약 **2.5M 스텝** (v5보다 대폭 빠름).

### v6 Resume 시 아키텍처 불일치

네트워크 크기 변경 후 old best_model을 resume하면 텐서 크기 불일치 오류 발생.

**해결**: `SAC.load()`를 try/except로 감싸 실패 시 자동 fresh start:
```python
try:
    model = SAC.load(resume, env=vec_env, ...)
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None  # 아래에서 새 모델 생성
```

---

## Phase 13 — 과탐색 역설과 적응형 ent_coef floor

### 9.83cm 스테이지 재정체

v6에서 빠르게 9.83cm에 도달했지만 이후 **13M+ 스텝** 동안 성공률 59~66% 횡보, 85% 미달.

### 핵심 진단: ent_coef_loss ≈ -8

지속적으로 관찰된 `ent_coef_loss ≈ -8`(크게 음수)은 SAC optimizer가 ent_coef를 **대폭 줄이고 싶다**는 신호.
현재 정책 엔트로피가 target(-1.0)보다 훨씬 높다 = "SAC 스스로 이제 exploit해야 한다고 판단하는 상태".

그런데 floor 0.15가 이를 막고 있어 **두 가지 탐색 강화가 동시 작동**:
1. floor 0.15: ent_coef를 강제로 높게 유지
2. action_noise σ=0.1: 롤아웃에 추가 노이즈

9.83cm 정밀도 달성에는 일관된 fine-grained exploit이 필요한데, 강제 탐색이 오히려 85% 도달을 방해.

### 이전에 이 스테이지가 더 쉬웠던 이유

git 이력 확인 결과, 구 버전의 advance threshold는 **70%**였음.
현재 성공률(60~66%)은 구 버전이라면 이미 통과 조건 충족.
즉 네트워크/탐색 설정 문제라기보다는 **더 엄격해진 advance 조건**과 **과탐색의 충돌**이 핵심.

### 해결: 적응형 ent_coef floor

```
평상시: base_floor = 0.05  ← SAC가 원하면 exploit 가능
부스트: boost_floor = 0.15 (2,000 에피소드 지속)
  트리거 1 — 커리큘럼 전진 시: 새 스테이지에서 재탐색 필요
  트리거 2 — 정체 감지 시: 10 윈도우(= 5,000 에피소드) 연속 성공률 2%p 이상 개선 없으면 자동 부스트
```

**핵심 아이디어**: exploit으로 수렴해 85% 달성 → 커리큘럼 전진 → 자동 re-explore → 다시 수렴.
v6에서 고착된 "과탐색으로 인한 정밀도 부족" 문제와, v5에서 겪은 "탐색 부족으로 인한 로컬 미니멈" 문제를 모두 방지.

```python
# 정체 10 윈도우 → boost
if self._windows_no_improvement >= self.stagnation_windows:
    self.ent_floor_callback.boost("정체 감지")

# 커리큘럼 전진 → boost + best_rate 리셋
if rate >= 85.0:
    self._set_curriculum(...)
    self.ent_floor_callback.boost("커리큘럼 전진")
    self._best_rate = 0.0
```

### 교훈: 탐색 강화와 수렴 허용은 동시에 달성 불가

- 항상-높은 floor: 수렴 방해 → 정밀도 부족
- 항상-낮은 floor: 로컬 미니멈 → 탈출 불가
- **적응형 floor**: "필요할 때만 탐색"이 올바른 접근

---

---

## Phase 14 — 커리큘럼 전진 부스트 역효과

### 현상

Phase 13에서 설계한 "커리큘럼 전진 시 boost" 로직이 오히려 성공률 정체를 유발.

**붕괴 패턴**:
```
성공률 85% 달성 → 커리큘럼 전진 → boost 발동 (ent_coef 0.15 강제)
→ 새 스테이지에서 정밀도 부족으로 성공률 급락
→ 정체 감지 → 부스트 반복
→ exploit 기회 없이 계속 탐색 → 성공률 50% 고착
```

**핵심 모순**: advance 직후 새 스테이지는 기존 정책이 "약간만 더 정밀하면 되는" 상태.
이때 ent_coef를 높이면 정밀도가 떨어져 성공률이 더 낮아짐.

### 해결

커리큘럼 전진 시 boost **제거**. 정체 boost만 유지.

```python
# 커리큘럼 전진 → boost 없이 카운터 리셋만
if self.ent_floor_callback is not None:
    self._best_rate = 0.0
    self._windows_no_improvement = 0
```

**교훈**: 커리큘럼 전진 직후는 "새 어려운 문제"가 아니라 "기존 정책 + 약간 더 타이트한 기준"이다.
재탐색이 아니라 추가 수렴이 필요한 시점.

---

## Phase 15 — 보상 가중치 불균형 (angle weight 0.3 → 1.0)

### 문제

`reward = -(pos_dist + 0.3 × angle_error)` 설계에서 정책이 위치만 최적화하고 자세를 무시.

- 7.84° 스테이지에서 1.5M+ 스텝 정체
- 성공률 50% 전후 진동 (성공 조건 = 위치 AND 자세 동시 만족)
- ent_coef, 커리큘럼 조정으로 해결 불가 → 보상 함수 문제로 진단

### 해결

```python
reward = -(pos_dist + 1.0 × angle_error)  # 0.3 → 1.0
```

**결과**: 7.84° 스테이지를 ~1.5M 스텝 만에 돌파 (이전 무한 정체).

**교훈**: HER 재라벨링 시 `compute_reward()`도 같은 가중치를 사용하므로, 가중치 불균형은 HER 학습 신호 전체에 영향을 미친다. 자세 정확도가 성공 조건에 포함된다면 보상에서도 동등하게 반영해야 한다.

---

## Phase 16 — 랜덤 단일 축 커리큘럼 전진

### 문제

기존: 성공률 ≥85% 시 pos + ori 동시 ×0.8 + init_range ×1.5 (모든 축 동시 상향).

이 설계의 문제:
- 전진 한 번에 세 변수가 동시에 어려워짐 → 효과적 난이도 점프가 복합적
- 어느 변수 때문에 성공률이 떨어졌는지 파악 불가
- 정책이 한 축은 충분히 학습했는데 다른 축도 동시에 어려워지면 낭비

### 해결

성공률 ≥85% 시 `pos / ori / obs / init_range` 중 **하나를 랜덤 선택**해 난이도 상승:

```python
options = ["pos", "ori"]
if self.max_obs_count < MAX_OBSTACLES:
    options.append("obs")
if self.init_range < RobotArmEnv.INIT_RANGE_MAX:
    options.append("init_range")
choice = np.random.choice(options)
```

**교훈**: 커리큘럼이 단순할수록 진단이 쉽다. 동시 다변수 조정은 어느 변수가 병목인지 숨긴다.

---

## Phase 17 — 정밀도 구간 ent_coef 재설계

### 문제 (4.18cm / 6.27° / obs 10 스테이지)

base_floor=0.05 유지 상태에서 4.18cm 스테이지 약 900K 스텝 이상 성공률 64~75% 진동.

**진단**:
- `ent_coef_loss ≈ -4.5` 지속: SAC optimizer가 ent_coef를 더 낮추고 싶어 함
- floor 0.05가 막고 있어 정밀 제어에 필요한 결정론적 행동이 제한됨
- `target_entropy=-1.0`: 정책 엔트로피 목표가 너무 높아 precision exploit 방해

**멀티 오브젝티브 충돌**: pos 정밀도 + ori 정밀도 + 장애물 10개 회피 동시 요구 → 그래디언트 간섭.

### 해결

```
ent_coef floor 제거 (base_floor=0.0): SAC가 원하는 만큼 낮출 수 있도록 허용
target_entropy -1.0 → -3.0: 더 결정론적인 정책 유도
정체 boost 복원: N 윈도우 개선 없으면 일시적 탐색 강화로 로컬 미니멈 탈출
```

**교훈**:
- `ent_coef_loss`가 지속적으로 음수(-4~-8)이면 SAC가 "이제 exploit이 더 필요하다"는 신호를 보내는 것. floor가 이를 막고 있다면 floor를 낮춰야 한다.
- precision task에서 `target_entropy`는 더 낮게(-3.0 이하) 설정하는 것이 적절.
- 단, floor 없이 두면 로컬 미니멈 위험이 있으므로 정체 감지 시 일시 boost는 유지.

---

## 현재 모델 상태

| 항목 | 값 |
|------|-----|
| 알고리즘 | SAC + HER (Torque) |
| 총 스텝 | ~7.5M (현재 run 기준) |
| pos threshold | 4.18cm |
| ori threshold | 6.27° |
| init_range | ±180° |
| 장애물 | 10/10 |
| ent_coef floor | 없음 (base_floor=0.0) / boost 0.15 (정체 시) |
| target_entropy | -3.0 |
| action_noise | N(0, 0.1) |
| net_arch | [512, 512, 512] |
| n_envs | 16 |

---

## 핵심 교훈 요약

1. **커리큘럼 변수와 목표를 분리하지 마라**: 타겟 분포와 성공 기준이 별도로 움직이면 학습이 어느 쪽을 향하는지 알기 어려움. `success_threshold` 하나로 통합이 효과적.

2. **FK를 쓸 거면 위치와 자세를 함께**: FK 위치만 맞추고 자세는 랜덤으로 하면 도달 불가 목표가 생성됨.

3. **보상 변경은 점진적으로**: 기존 모델에 새 보상항 즉시 추가 시 학습 붕괴. 커리큘럼으로 단계적 도입.

4. **ent_coef 자동 조절 믿지 마라**: SAC의 자동 엔트로피 조절은 탐색이 충분한 경우에도 ent_coef를 0에 가깝게 낮춤. 최솟값 클램핑 필수.

5. **Resume 시 warmup 필수**: replay buffer 재충전 기간 동안 curriculum이 역행하는 것을 막아야 함.

6. **±180° init_range에서 resume = 높은 실패 확률**: 커리큘럼 상태가 좋지 않으면 fresh start가 더 빠름.

7. **PPO보다 SAC+HER이 이 태스크에 압도적**: HER의 goal resampling이 희소 보상 문제를 효과적으로 해결.

8. **learning_starts 재설정이 대규모 모델에서 치명적**: resume 시 `learning_starts = num_timesteps + 30,000`이 되면 수백만 스텝 동안 정책 업데이트 없이 랜덤 액션만 실행. curriculum이 이 기간에 역행하면 회복 불가. resume 전 best_model 성능과 curriculum 상태의 정합성 반드시 확인.

9. **advance 조건은 스테이지별 난이도에 맞게 설정**: 90%는 obs 10개 + tight threshold 조합에서 달성 불가 수준. 85%도 6.29cm 스테이지에서 2.74M 스텝 소요. 모델이 구조적 한계에 부딪히면 advance 조건 완화를 고려.

10. **스테이지가 타이트해질수록 소요 스텝이 기하급수적으로 증가**: obs 0→10 구간 ~3.57M, 6.29cm 단일 스테이지 ~2.74M, 5.03cm 스테이지 ~24M 이상 추정. 100M 스텝 기준으로는 2~3cm 수준이 현실적 도달점.

11. **탐색과 수렴은 동시에 달성 불가 — 적응형 조절이 답**: 항상-높은 ent_coef floor는 fine-grained exploit을 방해해 높은 advance threshold(85%)에서 정체를 유발. 항상-낮은 floor는 로컬 미니멈 고착. `ent_coef_loss`가 크게 음수(-8)인데 floor가 이를 막고 있다면 "SAC가 exploit 신호를 보내고 있다"는 의미로 해석하라.

12. **advance threshold와 탐색 설정은 연동해서 고려**: threshold가 올라갈수록 폴리시가 더 일관된 precision이 필요하다. threshold를 높이면서 동시에 강제 탐색도 높이면 두 목표가 서로 충돌한다. 한 번에 하나씩 바꿔야 효과를 추적할 수 있다.

13. **아키텍처 변경 후 resume은 try/except로 보호**: ObstacleAwareExtractor나 net_arch 크기 변경 시 기존 best_model과 텐서 크기가 맞지 않아 로드 실패. 코드 수준에서 실패를 gracefully 처리해 자동 fresh start로 폴백하는 구조가 안전.

14. **커리큘럼 전진 직후 부스트는 역효과**: advance 이후 새 스테이지는 "조금 더 타이트한 기준"이므로 재탐색이 아닌 추가 수렴이 필요. 이때 ent_coef를 강제로 올리면 정밀도가 떨어져 성공률이 더 낮아지는 역설 발생. 정체 감지 시 boost는 유효하지만, advance 트리거 boost는 제거할 것.

15. **보상 가중치 불균형은 HER 전체를 오염시킨다**: HER 재라벨링 시 `compute_reward()`로 보상을 재계산하므로, 가중치 불균형이 있으면 재라벨된 경험 전체에 왜곡된 학습 신호가 전달된다. 자세 오차가 성공 조건에 포함되어 있다면 보상에서도 동등한 비중을 줘야 한다 (angle weight 0.3 → 1.0).

16. **다변수 커리큘럼 동시 전진은 병목 파악을 어렵게 한다**: pos/ori/obs를 동시에 어렵게 만들면 어느 변수 때문에 성공률이 하락했는지 알 수 없다. 랜덤 단일 축 전진으로 변경하면 각 스테이지의 병목 변수를 관찰할 수 있다.

17. **precision 구간에서 target_entropy는 낮게**: 정밀도 요구가 높아질수록 SAC는 더 결정론적인 정책을 필요로 한다. `ent_coef_loss`가 지속적으로 음수면 target_entropy를 낮추고(-3.0 이하) floor를 제거해 SAC가 원하는 만큼 결정론적으로 행동하게 하라. 단, 정체 감지 시 일시적 boost는 로컬 미니멈 탈출용으로 유지.
