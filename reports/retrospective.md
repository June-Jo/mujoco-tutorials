# 학습 회고 — 구 커리큘럼의 구조적 문제점

> 작성 기준: Position Run 1~26 (~52M 스텝), Torque Run 1~22 (~44M 스텝)
> 공통 결론: 하루 넘게 학습했음에도 불구하고 성공 threshold가 10cm 수준에 고착됨.

---

## 1. 커리큘럼 설계의 구조적 천장 (가장 근본적인 원인)

### 구 커리큘럼 구조
- 타겟 위치: **현재 EE 위치에서 반경 `max_r` 이내의 임의 점**
- `max_r = CURRICULUM_MIN_RADIUS + (MAX - MIN) × level`
  - level 0.00 → max_r = 8cm
  - level 0.01 → max_r = 10.5cm (+2.5cm)
  - level 1.00 → max_r = 260cm
- 커리큘럼: level ≥40% 성공 시 +0.01, <10% 시 -0.01

### 왜 level 0.01을 넘지 못했는가

**level 0.00 학습**: EE 중심 반경 8cm 타겟 → 시작 자세가 fixed zero이므로 EE 위치도 거의 고정 → 실질적으로 특정 공간의 작은 영역을 반복 학습하는 것과 동일.

**level 0.01 진입 시 즉각 붕괴**:
- joint 랜덤화 범위: `±level × JOINT_RANGES` = level 0.01 → ±0.063 rad (±3.6°)
- 6관절 복합 시 EE 불확실성 최대 수십 cm
- EE 중심 8~10cm 타겟이 joint 노이즈로 인한 EE 불확실성보다 작아짐
- 결과: 타겟이 실질적으로 "이미 EE 주변"에 없는 상태로 에피소드 시작 → 성공률 60%→10% 급락

**핵심 모순**: level 올라갈수록 joint 랜덤화도 강해져 EE 불확실성 증가 → 타겟 반경이 커지는 속도보다 불확실성이 더 빠르게 증가. 커리큘럼이 어려워지는 게 아니라 **다른 종류의 어려움**으로 바뀌는 구조.

---

## 2. Level-up/down 임계값 불안정

### 초기 설정 문제 (Run 3~5)
- level-up 임계값: 성공률 50% (너무 높아 달성 자체가 드묾)
- level-up 스텝: +0.05 (한 번에 max_r이 수 배로 증가)
- Run 4 예시: level 0.05 도달 후 max_r 0.08m → 0.21m로 2.6배 급증 → 즉각 붕괴

### 조정 후에도 반복된 문제 (Run 6~26)
- level-up: +0.01 (성공률 40%), level-down: -0.01 (성공률 15%→10%)
- level 0.00 ↔ 0.01 사이를 평균 500~1000 에피소드마다 진동 (Position 기준 6회 이상 반복)
- level-down 임계값을 10%로 완화 (Run 20~26)해도 level 0.01 장기 유지는 가능했으나 0.02 진입 불가

---

## 3. Value function 수렴 함정

**Position Run 20 관찰**:
- level 0.01에서 4000 에피소드 장기 유지 성공
- 그러나 `explained_variance ≈ 0.98`, `value_loss ≈ 0`
- Value function이 너무 잘 맞춰지면 policy gradient 신호가 약해짐
- 결과: policy가 개선되지 않고 현상 유지 → 결국 Ep 4500에서 level-down

**원인**: 좁은 target 분포(EE 중심 8~10cm) + 반복 학습 → 가치 함수가 과도하게 특화됨.

---

## 4. 선형 LR 감소 스케줄 (Run 5)

- `learning_rate = 3e-4 × (1 - progress)` 사용
- 820k 스텝 시점(총 1M 기준) → LR ≈ 5.4e-5
- 커리큘럼 level-up 직후 새 난이도 적응에 필요한 LR이 부족해 학습 불가
- **조치**: LR 상수 3e-4로 변경 (Run 6 이후 유지)

---

## 5. JOINT_RAND_MIN 실험 실패 (Position Run 10~18)

**의도**: 미세 노이즈를 추가해 robustness 향상

**결과**:
- `JOINT_RAND_MIN=0.01`: 효과 없음, 성능 오히려 악화
- `JOINT_RAND_MIN=0.02`: level 0 학습 자체 방해 (Ep 500에서 16~21%)
- `JOINT_RAND_MIN=0.005`: level 0.00에서 fresh start 시 1M 스텝에도 24% 고착
- Torque에서는 level 0.02 이상(joint 랜덤화 ±0.126 rad)에서 JOINT_RAND_MIN이 희석되어 무관

**교훈**: position 제어는 초기 학습 단계에서 EE 불확실성에 매우 민감. 외부 노이즈 추가가 초기 수렴을 방해.

---

## 6. Position vs Torque 비대칭

| 항목 | Position | Torque |
|------|----------|--------|
| 최종 커리큘럼 level | 0.01 (max_r≈10cm) | 0.02~0.05 (max_r≈13~21cm) |
| 최고 eval | ~+2.67 | +5.78 (Run 22) |
| level-up 속도 | 느림 (1500ep) | 빠름 (500ep) |
| level 안정성 | 낮음 (잦은 진동) | 높음 (9회 연속 유지) |

**추정 원인**: Position 제어는 Δθ action으로 점진적 이동 → 탐색 폭이 제한적. Torque 제어는 더 자유로운 궤적 탐색 가능 → 다양한 타겟 분포에 더 강건.

---

## 7. 결론 및 새 커리큘럼 설계 근거

### 핵심 문제 요약
> EE 중심 작은 반경으로 시작하는 커리큘럼은, joint 랜덤화와 결합될 때 구조적으로 level 0.01을 돌파할 수 없다. 하루 이상, 50M 스텝 이상을 학습해도 실질적으로 10cm 반경 문제만 반복 학습한 셈이다.

### 새 커리큘럼 설계 (Run 27 position / Run 23 torque부터 적용)

| 항목 | 구 설계 | 새 설계 |
|------|---------|---------|
| 시작 자세 | fixed zero (또는 약한 랜덤) | 완전 임의 (전체 JOINT_RANGES) |
| 타겟 위치 | EE 중심 반경 내 임의 점 | FK 기반 전체 workspace 임의 위치 |
| 커리큘럼 변수 | `curriculum_level` (0→1, max_r 결정) | `success_threshold` (30cm→0.1mm) |
| level-up 조건 | 성공률 ≥40% → +0.01 | 성공률 ≥70% → threshold ×0.5 |
| level-down 조건 | 성공률 <10% → -0.01 | 성공률 <20% → threshold ×2 |
| 최종 목표 | level 1.0 (max_r 260cm) | threshold 0.1mm |

**새 설계의 장점**:
1. 처음부터 전체 workspace를 학습 → EE 반경에 의한 구조적 천장 없음
2. 성공 기준이 곧 커리큘럼 변수 → 직관적이고 실제 목표(정밀도)와 직결
3. threshold ×0.5 조정으로 점진적 난이도 증가 (기하급수적 → 자연스러운 로그 스케일)
4. 70% 기준으로 충분히 학습됐을 때만 다음 단계 진입 → 성급한 level-up 방지
