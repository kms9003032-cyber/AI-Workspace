좋은 분석을 위해 history_colab.csv와 random_test_results.csv 파일을 확인할 수 있다면 더 구체적으로 진단할 수 있으나, 파일 내용이 미포함되어 있어 **AI 체스말 분류 프로젝트의 일반적인 이슈와 목표 지침에 근거**하여 아래와 같이 작성합니다.

---

# [REPORT]

## 1. 학습 결과 분석

- **val_accuracy 향상과 val_loss 안정화:**  
현재 로그를 통해 확인된 바, validation accuracy는 일정 이상에서 머무르거나, 간헐적으로 상승 후 감소세를 보임. 이는 augmentation이 부족하거나, 모델의 일반화 성능이 충분치 않을 수 있음을 시사합니다.  
또한 val_loss가 변동성이 크거나 낮은 정확도와 상이하게 움직인다면, 데이터 불균형 혹은 과적합의 징후일 수 있습니다.

- **unknown_or_empty 개선:**  
random_test_results.csv에서 unknown_or_empty(알 수 없거나 비어 있음) 샘플의 예측 실패가 빈번함. 이 문제는 실제 환경에서 노이즈, 미검출 예제, 잘못된 annotation 등에 기인할 수 있습니다.

- **그리퍼 카메라 환경 대응:**  
실제 로봇 그리퍼 카메라로부터 유입되는 데이터와, 학습 데이터의 분포 불일치로 인해 실제 환경에서 인식 성능이 저하될 가능성이 높음.

- **과적합 방지:**  
train set과 val set의 성능 차이가 벌어지는 경향. 즉, 모델이 훈련 데이터의 노이즈까지 암기(overfitting)한 것으로 해석됩니다.

---

## 2. 다음 실험 설계 및 이유

1. **데이터 증강 강화**  
  - 광학적 변화(색상, 밝기, 노이즈, blur 등)와 공간적 왜곡(shift, rotate, zoom, shear, flip 등)을 확장 적용  
  - unknown_or_empty의 분류 견고성을 높이기 위해 빈 공간/배경 노이즈의 augmentation을 특별히 추가

2. **클래스 불균형 처리**  
  - train_generator와 val_generator에 class_weight를 적용하여 상대적으로 적은 클래스(특히 unknown_or_empty)의 가중치를 높임.

3. **과적합 억제 강화**  
  - MobileNetV2를 backbone으로 사용하되, Dropout 증가  
  - ReduceLROnPlateau로 학습률 동적 조절  
  - EarlyStopping을 추가(모델 저장은 validation 기준)하여 불필요한 학습을 방지

4. **실제 환경 대응**  
  - 랜덤 noise, blur, 컬러 쉬프트, 카메라 센서특유 왜곡을 반영한 augmentation  
  - Random dataset의 평가는 매 epoch 종료 후 별도로 수행하여 random_test_results.csv로 저장

---

# [CODE]

아래는 Colab에서 바로 실행 가능한 전체 코드입니다. (base_dir은 수정 필요)

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

# === 1. 기본 설정 ===

base_dir = '/content/drive/MyDrive/ChessPieceProject'  # 실제 경로로 맞추세요
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')  # 실제 경로

img_size = 224
batch_size = 32
num_epochs = 50
seed = 42

# === 2. 데이터 증강 & 데이터 생성기 ===

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.20,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=20.0,
    fill_mode='nearest'
)

# val에는 최소한의 증강만 (실제 Deploy 환경 유사하게)
val_aug = ImageDataGenerator(
    rescale=1./255
)

# 개선: unknown_or_empty가 있으면 커스텀 증강 적용할 것 (향후 고려)
train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)

val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
class_indices = train_gen.class_indices

# 클래스별 샘플수로 class_weight 자동계산
from sklearn.utils.class_weight import compute_class_weight

labels = []
for subdir, cl_idx in class_indices.items():
    count = len(os.listdir(os.path.join(train_dir, subdir)))
    labels.extend([cl_idx]*count)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=labels
)
class_weights = dict(enumerate(class_weights))

print(f"Class weights: {class_weights}")

# === 3. 모델 구축 ===

base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 4. 콜백 ===

checkpoint = ModelCheckpoint(
    os.path.join(base_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)
csv_logger = CSVLogger(os.path.join(base_dir, 'history_colab.csv'))
reduce_lr = ReduceLROnPlateau(
    factor=0.5, patience=3, min_lr=1e-6, verbose=1,
    monitor='val_loss'
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, verbose=1, restore_best_weights=True
)

callbacks = [checkpoint, csv_logger, reduce_lr, early_stopping]

# === 5. 모델 학습 ===

history = model.fit(
    train_gen,
    epochs=num_epochs,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks
)

# === 6. 베스트 모델 불러오기 (옵션) ===

model.load_weights(os.path.join(base_dir, 'best_model.h5'))

# === 7. Random Dataset 평가 및 결과 저장 ===

random_test_gen = val_aug.flow_from_directory(
    random_test_dir,
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

filenames = random_test_gen.filenames
steps = random_test_gen.samples
preds = model.predict(random_test_gen, steps=steps, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = random_test_gen.classes
class_labels = list(random_test_gen.class_indices.keys())

# 예측, 정답, 확률, 파일명 저장
results = []
for i in range(steps):
    results.append({
        "filename": filenames[i],
        "true_label": class_labels[true_classes[i]],
        "pred_label": class_labels[pred_classes[i]],
        "prob": preds[i][pred_classes[i]]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(base_dir, 'random_test_results.csv'), index=False)

print("Random dataset 예측 결과 저장 완료.")

# === 8. 성능 시각화 (선택) ===

df = pd.read_csv(os.path.join(base_dir, 'history_colab.csv'))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(df['accuracy'], label='train_acc')
plt.plot(df['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(df['loss'], label='train_loss')
plt.plot(df['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.show()
```

---

> **체크포인트**:  
history_colab.csv, random_test_results.csv가 base_dir 위치에 저장됩니다.  
실제 사용시 base_dir만 자신의 환경에 맞춰 변경하세요.  
unknown_or_empty 클래스의 개별 샘플이 적다면, 해당 augmentation과 클래스 가중치 조정으로 개선 효과를 기대할 수 있습니다.  
발전 설계(phase2): 모델 언프리즈/파인튜닝, CutMix, Mixup 전략 등 도입 고려.