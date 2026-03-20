# Position 제어 학습 이력

## Run 1 — 수동 종료 (보상 함수 문제)
- **스텝**: ~500k
- **종료 원인**: 보상 함수 설계 오류
  - `reward = -dist` 사용 → 거리 절댓값 기반 보상이라 스케일이 불안정하고, 다가가는 행동에 즉각적인 피드백 없음
  - potential shaping(`prev_dist - dist`)으로 교체 필요
- **조치**: 보상 함수를 `(prev_dist - dist) × 10` + 성공 보너스 + 생존 패널티로 변경

---

## Run 2 — 수동 종료 (환경 리팩터링)
- **스텝**: ~114k
- **종료 원인**: 코드 구조 개선 필요
  - `control_mode` 파라미터로 position/torque를 하나의 클래스에서 분기 처리 → 각 모드의 action 의미가 불명확
  - `RobotArmEnv` → `RobotArmBaseEnv` + `RobotArmPositionEnv` + `RobotArmTorqueEnv`로 분리
- **조치**: 베이스 클래스 + 서브클래스 구조로 리팩터링

---

## Run 3 — 에러 종료 (ImportError)
- **스텝**: 0 (시작 실패)
- **종료 원인**: `train.py`에서 `from robot_env import RobotArmEnv` 임포트 실패
  - 리팩터링으로 클래스명이 `RobotArmBaseEnv`로 변경됐는데 `train.py` 미반영
- **조치**: `from robot_env import RobotArmBaseEnv as RobotArmEnv`로 수정

---

## Run 4 — 수동 종료 (Curriculum 불안정)
- **스텝**: ~760k
- **주요 지표**: level 0.05 도달 후 2000ep 성공률 4.4%로 급락, eval reward -4.07
- **종료 원인**: Curriculum 상승 스텝(+0.05)이 너무 커서 level-up 시 성능 붕괴
  - max_r이 0.08m → 0.21m로 한 번에 2.6배 증가 → 정책이 새 난이도에 적응 못함
- **조치**: 상승 스텝 0.05 → 0.02, 하강 스텝 0.02 → 0.01로 조정

---

## Run 5 — 수동 종료 (LR 감소 + Curriculum 불안정)
- **스텝**: ~820k
- **주요 지표**: level 0.02 도달 후 2500ep 성공률 6.6%로 붕괴, eval -2.42
- **종료 원인**: 선형 LR 감소 스케줄(`3e-4 × f`)로 인해 820k 시점에 LR이 5.4e-05까지 떨어짐
  - Curriculum level-up 후 새 난이도 적응에 필요한 LR이 너무 낮아 학습 불가
  - +0.02 step에서도 동일한 성능 붕괴 패턴 반복
- **조치**: LR 상수(`3e-4`) 변경, 총 스텝 2M으로 증가

---

## Run 6 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 7)

---

## Run 7~11 — 복수 재시작 (Curriculum 진동 + 실험)
- **종료 원인**: level 0.00 ↔ 0.01 진동 구조적 문제 지속
  - JOINT_RAND_MIN=0.01 실험 (Run 10~11): 효과 없음, 성능 오히려 악화 → 코드 원복
  - level 0.01 진입 시마다 500ep 내 12~15%로 붕괴, level 0.00으로 복귀 패턴 반복
- **조치**: 원본 코드(JOINT_RAND_MIN 없음) 유지, fresh start (Run 12)

---

## Run 12 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval +2.67 (2M), ep_rew_mean +1.16
- **커리큘럼**: level 0.00 ↔ 0.01 진동 6회 반복
  - Ep 500: 40.4% → level 0.01 → Ep 1000: 12.0% → level 0.00
  - Ep 2000: 50.8% → level 0.01 → Ep 2500: 13.6% → level 0.00
  - Ep 3000: 43.6% → level 0.01 → Ep 3500: 14.4% → level 0.00
  - Ep 4500: 49.0% → level 0.01 → Ep 5000: 12.4% → level 0.00
  - Ep 5500: 37.0% @ level 0.00
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 13)

---

## Run 13 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -2.43 (2M), level 0.00 ↔ 0.01 진동 6회 반복
  - Ep 500/2001/3001/4001/5001: level 0.01 진입 (50~55%)
  - Ep 1001/2501/3501/4501/5501: 11~14%로 붕괴 → level 0.00
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 14)

---

## Run 14 — (이전 기록, 내용 없음)

---

## Run 15 — 수동 종료 (JOINT_RAND_MIN=0.02 과도)
- **스텝**: ~819k
- **주요 지표**: Ep 500 4연속 16~21% @ level 0.00
- **종료 원인**: JOINT_RAND_MIN=0.02가 level 0 학습 자체를 방해
- **조치**: JOINT_RAND_MIN=0.005로 축소 후 fresh start (Run 16)

---

## Run 16 — 수동 종료 (JOINT_RAND_MIN=0.005도 fresh start에서 부적합)
- **스텝**: ~1,015k
- **주요 지표**: Ep 500~2500 5연속 20~24% @ level 0.00 (level-up 불가)
- **종료 원인**: JOINT_RAND_MIN=0.005도 fresh start에서 level 0 학습 방해
  - 이전 fresh start(Run 12, JOINT_RAND_MIN 없음)은 Ep 500에 40.4% 달성했으나
  - 현재 run은 1M 스텝 이후에도 24% 수준 고착
- **조치**: JOINT_RAND_MIN=0.005 유지하되, best_model resume으로 재시작 (Run 17)
  - best_model이 level 0을 이미 학습 → 초기 40%+ 달성 기대

---

## Run 17 — 수동 종료 (level 0 고착, best_model resume 무효)
- **스텝**: ~671k
- **주요 지표**: Ep 500~1500 3연속 하락 (28.2%→25.8%→24.2%) @ level 0.00
- **종료 원인**: best_model이 JOINT_RAND_MIN=0.005(level=0에서 ±0.3° 노이즈)에 적응 실패
  - best_model은 JOINT_RAND_MIN 없이 학습 → 미세 노이즈에 민감
  - curriculum_level.txt 없어 level 0.00에서 시작 → JOINT_RAND_MIN이 전면 적용
- **조치**: curriculum_level.txt=0.01 수동 설정 후 재시작 (Run 18)
  - level 0.01에서는 joints randomization ≈ ±0.063 rad → JOINT_RAND_MIN 희석
  - best_model이 더 익숙한 범위에서 시작 가능

---

## Run 18 — 수동 종료 (JOINT_RAND_MIN=0.005 position에서 구조적 실패 확정)
- **스텝**: ~1,300k
- **주요 지표**: Ep 500~1500 (22.2%→26.4%→24.0%) @ level 0.01 — 40% 달성 불가
- **종료 원인**: Run 16~18 3회 연속 JOINT_RAND_MIN=0.005가 position level 0/0.01 학습 방해
  - Torque는 level 0.02(joints ±0.126 rad)에서 JOINT_RAND_MIN 희석되어 무관
  - Position은 level 0.01(joints ±0.063 rad)에서도 40% 미달 고착
- **조치**: JOINT_RAND_MIN=0.005 → 0.0으로 복원, curriculum_level.txt 삭제 후 재시작 (Run 19)

---

## Run 19 — 수동 종료 (level 0.00↔0.01 진동 재발)
- **스텝**: ~820k
- **주요 지표**: Ep 500: 51.2% → level 0.01, Ep 1000: 11.8% → level 0.00 (붕괴)
- **종료 원인**: JOINT_RAND_MIN=0 복원으로 level 0 학습은 해결됐으나 level-up 직후 붕괴 패턴 재발
  - level-down 임계값 15%가 너무 민감 (11~14% 수준에서 level-down 반복 발생)
- **조치**: level-down 임계값 15%→10%로 완화 (train.py) 후 재시작 (Run 20)

---

## Run 20 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -2.42 (2M), ep_rew_mean -3.67, level 0.00 (Ep 4500에서 첫 level-down)
- **커리큘럼**:
  - Ep 500: 55.8% → level 0.01
  - Ep 1000~4000: 11~13% @ level 0.01 (10% threshold로 level-down 방지, 4000 에피소드 유지)
  - Ep 4500: 9.8% → level 0.00 ↓ (10% threshold 첫 발동)
- **특이사항**: level 0.01 장기 유지 성공했으나 value function 수렴 함정(explained_variance≈0.98, value_loss≈0)으로 policy gradient 신호 약화 → 회복 불가
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 21)

---

## Run 21 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -3.92 (2M), ep_rew_mean -9.56
- **커리큘럼**:
  - Ep 500: 57.0% → level 0.01
  - Ep 1000: 10.2%, Ep 1500: 13.0%, Ep 2000: 10.8% @ level 0.01
  - Ep 2500: 8.4% → level 0.00 (level-down)
  - Ep 3000: 40.0% → level 0.01 재진입
  - Ep 3500: 12.2%, Ep 4000: 12.8%, Ep 4500: 12.0% @ level 0.01 (역대 최장 3회 연속 유지)
- **특이사항**: level 0.01 Ep 3000~4500 1500 에피소드 연속 유지 — Run 21 최고 기록
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 22)

---

## Run 22 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -3.05 (2M), ep_rew_mean -9.13
- **커리큘럼**:
  - Ep 500: 56.6% → level 0.02 (최초 달성!)
  - Ep 1000: 8.0% → level 0.01 (level-down)
  - Ep 1500~2500: 13.6%/14.2%/13.2% @ level 0.01 (3회 연속 유지)
  - Ep 3000: 9.6% → level 0.00 (level-down)
  - Ep 3500: 34.6%, Ep 4000: 42.0% → level 0.01 재진입
  - Ep 4500: 13.2%, Ep 5000: 12.4% @ level 0.01 (2회 추가 유지)
- **특이사항**: level 0.01 총 6회 유지 (Ep 1500~2500 + 4000~5000), level 0.02 최초 진입
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 23)

---

## Run 23 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -2.43 (2M), ep_rew_mean -8.52
- **커리큘럼**:
  - Ep 500: 62.4% → level 0.02 ↑ (빠른 level-up)
  - Ep 1000: 9.8% → level 0.01 ↓
  - Ep 1500~4500: **8회 연속** 10.4~13.6% @ level 0.01 유지
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 24)

---

## Run 24 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -0.511 (best), -2.38 (2M), ep_rew_mean -2.6
- **커리큘럼**:
  - Ep 500: 66.6% → level 0.02 ↑
  - Ep 1000: 7.4% → level 0.01 ↓
  - Ep 1500~4000: **7회 연속** 11.8~13.2% @ level 0.01 유지
  - Ep 4500: 9.6% → level 0.00 ↓ (마지막 체크에서 level-down)
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 25)

---

## Run 25 — 정상 완료 (2M 스텝)
- **스텝**: 2,015,232
- **최종 지표**: eval -2.36 (2M), -3.73 (2,010k), ep_rew_mean -8.43
- **커리큘럼**:
  - Ep 500: 66.8% → level 0.01 ↑ (level 0.00에서 시작, Run 24 끝에서 level-down)
  - Ep 1000~5000: **9회 연속** 10.8~14.6% @ level 0.01 유지 — **역대 신기록 🏆**
- **특이사항**: 이전 최고 기록(Run 23 8회) 경신. level 0.02 진입은 없었으나 level 0.01 최장 유지
- **종료 원인**: 2M 스텝 예산 소진 (정상 완료)
- **조치**: best_model로 재시작 (Run 26)

---

## Run 26 — 조기 종료 (커리큘럼 재설계)
- **스텝**: ~1,050k (수동 종료)
- **최종 지표**: level 0.00 고착 (Ep 2500: 23.6%)
- **종료 원인**: 커리큘럼 구조적 한계 확인 — 26회 Run (~52M 스텝) 동안 level 0.01을 넘지 못함
  - level 0.00 max_r=8cm에서 학습 → level 0.01 진입 시마다 즉각 붕괴 반복
  - 근본 원인: EE 중심 반경 기반 커리큘럼 + joint 랜덤화 상호작용
- **조치**: 커리큘럼 전면 재설계 후 재시작 (Run 27)
  - FK 기반 완전 임의 타겟 (전체 workspace)
  - success_threshold 기반 커리큘럼 (30cm → 0.1mm, ×0.5/×2 조정)

---

## Run 27 — 진행 중 (새 커리큘럼)
- **시작 스텝**: 0 (best_model resume, 새 커리큘럼)
- **변경 사항**:
  - 타겟 샘플링: EE 중심 반경 → FK 기반 전체 workspace 임의 위치
  - 시작 자세: 완전 임의 (전체 JOINT_RANGES)
  - 커리큘럼 변수: curriculum_level → success_threshold (30cm 시작)
  - level-up: 성공률 ≥70% → threshold ×0.5 / level-down: <20% → threshold ×2
