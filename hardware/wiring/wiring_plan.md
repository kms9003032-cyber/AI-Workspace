# Wiring Plan

## 현재 상태
아직 모든 선과 부품이 도착하지 않아 최종 배선은 진행하지 못함.

## 목표
Arduino MEGA 2560, RAMPS 1.4, PCA9685, 모터 드라이버, 전원공급장치, 컨버터, 카메라를 연결하는 배선 구조를 정리한다.

## 우선 확인할 것
- 전원공급장치 24V 출력
- 스테퍼 모터 드라이버 연결
- 서보모터용 PCA9685 연결
- 24V → 7V 컨버터 연결
- 24V → 5V 컨버터 연결
- Arduino MEGA 2560 + RAMPS 1.4 연결
- Python ↔ Arduino Serial 통신

## 현재 전략
최종 배선 전, 먼저 USB 연결을 통해 Python에서 Arduino로 명령을 보내고 Arduino가 명령을 수신하는지 확인한다.
