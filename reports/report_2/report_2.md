# 응용기계설계 2차 보고서 (참고 및 기록용)

> 본 문서는 실제 제출했던 2차 보고서를 기반으로,
> 현재 프로젝트 진행 상황과 알고리즘 구조를 기록 및 관리하기 위해
> Markdown 형태로 재구성한 참고용 문서이다.

---

# 1. 프로젝트 목표

본 프로젝트의 목표는 체스판을 인식하고,
체스말의 상태 및 위치를 판단한 뒤,
실제 로봇이 체스말을 자동으로 이동 및 정리하는
AI 기반 체스 자동화 시스템을 구현하는 것이다.

단순 체스 AI 구현이 아니라,

Vision → 판단 → 좌표 변환 → 모터 제어

까지 연결되는 실제 피지컬 AI 시스템 구현을 목표로 한다.

---

# 2. 프로젝트 전체 시스템 구조

## 전체 흐름

사용자 수 진행  
↓  
상부 카메라 체스판 촬영  
↓  
체스판 변화 감지  
↓  
게임 엔진(Stockfish) 전달  
↓  
컴퓨터 수 계산  
↓  
translation.py 좌표 변환  
↓  
Arduino 좌표 전달  
↓  
모터 구동  
↓  
그리퍼 동작  
↓  
체스말 이동  
↓  
상부 카메라 재확인

---

# 3. 하드웨어 구조

## 최종 구조

![Final Structure](images/final_structure.jpg)

현재 시스템은 갠트리 기반 XY 이동 구조를 사용한다.

- XY 리니어 이동
- Z축 승강 구조
- Pan/Tilt 그리퍼
- 상부 카메라
- 그리퍼 카메라

를 기반으로 체스말을 이동한다.

## 지브 구동부

![Jib Drive](images/jib_drive.jpg)

지브 구동부는:

- 2060 알루미늄 프로파일
- NEMA23 스테퍼 모터
- 평기어 기반 감속 구조

를 사용한다.

8:1 기어비를 적용하여
출력을 증가시키는 구조를 사용하였다.

---

## 트롤리 이동부

![Trolley System](images/trolley_system.jpg)

트롤리 이동부는:

- GT2 벨트
- 리니어 레일
- NEMA17 모터

기반 구조를 사용한다.

상부 레일을 따라 이동하며
체스판 위치를 탐색한다.

---

## Z축 이동부

![Z Axis](images/z_axis_system.jpg)

Z축은:

- 리드스크류
- 리니어 액추에이터
- 리니어 레일

기반 구조를 사용한다.

그리퍼 승강을 담당한다.

---

## 그리퍼 구동부

![Gripper](images/gripper_system.jpg)

그리퍼는:

- MG996R 서보모터
- Pan/Tilt 구조
- 평행 그리퍼 방식

을 사용한다.

현재 CAD에는 임시 그리퍼 모델이 포함되어 있으며,
실제 사용 예정 그리퍼는 별도로 관리한다.

---

# 4. 구조 해석

## 응력 해석

![Stress Analysis](images/stress_analysis_1.jpg)

외팔보 구조 특성을 기반으로
최대 응력 위치를 분석하였다.

최대 응력은
지브 연결부에서 발생하였다.

---

## Von Mises 응력 및 안전계수

![Stress Analysis 2](images/stress_analysis_2.jpg)

최대 응력:
124.06 MPa

안전계수:
2.02

로 계산되었다.

일반 기계 설계 기준인
1.5~2.0 수준을 만족하였다.

---

## 변위 해석

![Displacement](images/displacement_analysis.jpg)

최대 변위:
1.32 mm

수준으로 나타났으며,
실제 구동 환경에서도
구조 안정성에는 큰 문제가 없을 것으로 판단하였다.

---

# 5. 체스말 인식 AI

## 목표

그리퍼 카메라를 통해
체스말 종류를 근거리에서 재분류한다.

상부 카메라와 별도로
정확한 기물 분류를 수행하기 위해 구현하였다.

# 6. 데이터셋 구조

![Dataset Structure](images/dataset_structure.jpg)

현재 데이터셋은 다음 구조로 관리한다.

```text
train/
validation/
raw_dataset/
results/
models/
```

---

## train

실제 학습 데이터 저장 폴더.

---

## validation

검증 데이터 저장 폴더.

---

## raw_dataset

새롭게 촬영한 테스트용 원본 이미지 저장 폴더.

---

## results

AI가 자동 분류한 결과 저장 폴더.

※ 중요:

results 폴더의 데이터를  
검수 없이 재학습에 사용하면 안 된다.

반드시 사람이 직접 확인 후  
train/validation 폴더로 이동해야 한다.

---

## models

학습 완료된 .keras 모델 저장 폴더.

---

# 7. AI 학습 구조

## 데이터 증강

![Augmentation](images/augmentation_code.jpg)

현재 다음 증강 기법을 사용한다.

- rotation_range
- zoom_range
- width_shift_range
- height_shift_range
- brightness_range

---

## 모델 구조

![MobileNet](images/mobilenet_structure.jpg)

현재 MobileNetV2 기반 Transfer Learning 구조를 사용한다.

TensorFlow + Keras 기반으로 구현하였다.

---

## 학습 결과

![Training Result](images/training_result.jpg)

현재 validation accuracy는
약 90% 이상 수준이다.

최고 정확도는
약 92% 수준까지 도달하였다.

---

# 8. 현재 AI 문제점

## Empty Detection 문제

현재 빈 공간(empty)을
기존 기물로 잘못 분류하는 문제가 존재한다.

예시:

- empty → white_pawn

으로 분류되는 문제 발생.

원인:

- empty 데이터 부족
- unknown 상황 학습 부족

으로 판단된다.

# 9. 상부 카메라 알고리즘

## 체스판 인식

![Camera Grid](images/camera_grid_system.jpg)

상부 카메라는:

- 체스판 격자 생성
- 변화 감지
- 사용자 수 판별

역할을 담당한다.

현재는:
0 / 1 기반 방식으로
체스말 존재 여부를 판단한다.

---

# 10. 게임 엔진

현재 Stockfish 기반 체스 엔진을 사용한다.

상부 카메라가 감지한 변화값을
체스 좌표로 변환 후
게임 엔진에 전달한다.

예시:

```text
e2e4
e7e5
```

---

# 11. translation.py 좌표 변환

![Translation](images/translation_system.jpg)

translation.py는:

체스 좌표  
→ 실제 mm 좌표

로 변환하는 역할을 수행한다.

예시:

```text
e2 → (x, y)
e4 → (x, y)
```

이후 Arduino에 전달 가능한 형태로 변환된다.

---

# 12. Arduino 및 모터 제어

현재 사용 예정:

- Arduino MEGA2560
- RAMPS 1.4
- PCA9685
- NEMA23
- NEMA17
- NEMA11

구조를 사용한다.

향후 Python ↔ Arduino 간 통신을 통해
실시간 모터 제어를 진행할 예정이다.

---

# 13. 현재 진행 상황

## 완료된 부분

- 전체 CAD 설계
- 구조 해석
- AI 학습 구조 구현
- 체스 엔진 구현
- 상부 카메라 구조 구현
- translation.py 구현
- 데이터셋 구축

---

## 현재 문제점

- 조명 변화 민감성
- 카메라 흔들림 문제
- empty detection 정확도 부족
- 실제 하드웨어 연동 미완료

- # 14. 향후 목표

## 단기 목표

- 실제 하드웨어 조립
- Arduino 연결
- 모터 제어 테스트
- 그리퍼 테스트
- 실제 체스말 이동 구현

---

## 중기 목표

- 체스 자동 정리 알고리즘
- 실시간 오류 보정
- 카메라 안정화
- empty 데이터 추가 학습

---

## 최종 목표

사용자와 실제 체스를 둘 수 있으며,  
게임 종료 후 체스말 자동 정리까지 가능한  
AI 기반 피지컬 체스 로봇 구현.

---

# 15. 현재 프로젝트 병목 현상

현재 프로젝트의 가장 큰 병목은
Vision 시스템과 실제 하드웨어 연결 과정이다.

특히 다음 문제들이 존재한다.

- 좌표 변환 오차
- 카메라 흔들림
- 조명 변화 민감성
- 다중 기물 상황 처리
- 실제 그리퍼 충돌 가능성
- empty 데이터 부족
- Arduino ↔ Python 실시간 연결 미완성

현재는:
"AI 정확도 향상"보다

전체 시스템 연결 및
실제 동작 구현을 우선 목표로 한다.

---

# 16. 현재 프로젝트 방향성

본 프로젝트는 단순 체스 AI 프로젝트가 아니라,

Vision → 판단 → 좌표 변환 → 물리 제어

까지 연결되는
실제 피지컬 AI 시스템 구현을 목표로 한다.

따라서:

- 실제 연결 가능성
- 하드웨어 안정성
- 전체 시스템 흐름
- 예외 상황 처리
- 실제 구동 가능성

을 중요하게 고려한다.

---

# 17. 참고 사항

본 문서는
실제 제출했던 2차 보고서를 기반으로
Markdown 형태로 재구성한 참고용 문서이다.

현재 실제 프로젝트 진행 상황과
일부 차이가 존재할 수 있다.

최신 상태는:

- memory/project_state.md
- issues/current_issues.md
- roadmap/current_goal.md

파일을 기준으로 관리한다.

