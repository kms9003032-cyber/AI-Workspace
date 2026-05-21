# Daily Training Notes

## 목적

자동 학습 결과와 실험 내용을 기록하는 파일이다.

매 학습 cycle 이후:

- accuracy
- val_accuracy
- loss
- val_loss
- learning rate
- random test 결과
- 문제점

을 기록한다.

---

# 기록 예시

## 2026-05-21

### 결과

- accuracy:
- val_accuracy:
- loss:
- val_loss:
- learning_rate:

### 문제점

- unknown_or_empty 정확도 낮음
- empty → white_pawn 오분류 발생
- 조명 영향 존재

### 다음 목표

- empty 데이터 추가
- 그리퍼 카메라 데이터 추가
- random background 추가

---

# 장기 목표

향후 n8n 및 AI Agent가
이 파일을 읽고:

- 문제점 분석
- 자동 개선 방향 제안
- 다음 실험 추천

까지 수행하는 것을 목표로 한다.
