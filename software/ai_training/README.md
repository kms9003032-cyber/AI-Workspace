# AI Training Code Files

현재 AI 학습 관련 실제 코드 파일은 다음과 같이 관리한다.

```text
software/
└── ai_training/
    ├── README.md
    ├── train_model_colab.py
    ├── predict_and_sort.py
    ├── notes.md
    └── models/
```

---

# train_model_colab.py

## 역할

- MobileNetV2 기반 모델 학습
- train / validation 데이터 로딩
- 데이터 증강 적용
- 모델 저장
- accuracy / loss 확인

## 주요 기능

- ImageDataGenerator 사용
- TensorFlow / Keras 사용
- Transfer Learning 사용
- validation accuracy 확인
- best_model 저장

---

# predict_and_sort.py

## 역할

새로운 이미지들을 AI로 자동 분류 후
results 폴더에 정리하는 코드.

예시:

```text
raw_dataset/
↓
AI 자동 분류
↓
results/
    ├── white_pawn/
    ├── black_queen/
    └── unknown_or_empty/
```

---

# 중요 주의사항

results 폴더에 자동 분류된 이미지를 그대로 학습에 사용하면 안 된다.

반드시 사람이 직접 확인 후:

- train/
- validation/

폴더로 이동한 뒤 재학습해야 한다.

---

# 현재 AI 문제점

- validation accuracy 불안정
- loss 값 큼
- unknown_or_empty 데이터 부족
- 실제 체스판 환경 정확도 미검증
- 실제 그리퍼 카메라 환경 미검증
- 조명 변화 민감성 존재
- 그림자 영향 존재

---

# 앞으로 추가 예정 코드

## 예정 기능

- confusion matrix 분석
- per-class accuracy 분석
- 잘못 분류된 이미지 자동 저장
- confidence score 출력
- 실시간 카메라 테스트
- 실제 그리퍼 카메라 재학습

---

# 현재 AI 방향성

현재 프로젝트는:

"AI 정확도 극한 향상"

보다,

"실제 하드웨어와 연결 가능한 수준"

구현을 우선 목표로 한다.

즉:

- 실제 사용 가능성
- 실시간 처리 가능성
- 실제 환경 적응성

을 중요하게 고려한다.


---

# Local Automation System

## 목적

집 데스크탑에서 매일 자동으로 체스말 분류 AI를 학습하고,
학습 결과와 랜덤 이미지 테스트 결과를 저장하기 위한 시스템이다.

현재 목표는:

- validation accuracy 추적
- loss 추적
- random image 테스트
- unknown_or_empty 문제 분석
- 장기적인 자동 개선 구조 구축

이다.

---

# 자동 학습 실행 흐름

```text
run_training.bat 실행
↓
train_model_local.py 실행
↓
모델 학습
↓
training_results/history.csv 저장
↓
models/best_model.keras 저장
↓
evaluate_random_images.py 실행
↓
raw_dataset 랜덤 테스트
↓
training_results/random_test_results.csv 저장
```

---

# 필요한 로컬 폴더 구조

```text
software/ai_training/
├── dataset/
│   ├── train/
│   ├── validation/
│   └── raw_dataset/
├── models/
├── training_results/
├── train_model_local.py
├── evaluate_random_images.py
└── run_training.bat
```

---

# 실행 방법

Windows에서:

```text
run_training.bat
```

더블클릭.

---

# 중요 주의사항

results 또는 random_test에 자동 분류된 이미지를
그대로 학습에 사용하면 안 된다.

반드시 사람이 직접 확인 후:

- train/
- validation/

폴더로 옮긴 뒤 재학습해야 한다.

---

# 장기 목표

향후 목표:

- Windows 작업 스케줄러 기반 자동 학습
- 매일 00:00 자동 실행
- val_accuracy / loss 자동 분석
- random_test_results.csv 자동 분석
- n8n 기반 결과 분석
- GitHub 기반 자동 개선 제안
- AI Agent 기반 자동 연구 보조 시스템 구축

---

# 현재 우선 목표

현재는:

"완전 자동 AI"

보다,

"자동 학습 및 자동 분석 가능한 기반 구축"

을 우선 목표로 한다.
