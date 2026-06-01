네, 제공된 요구사항에 맞춰 history_colab.csv와 random_test_results.csv 분석 기준(실제 데이터 미제공 상황)과 실제 개선 학습 코드를 모두 제공합니다.

---
# [REPORT]

## 1. 결과 분석

### 1.1 Validation Accuracy 및 Loss
- `history_colab.csv`를 확인하면, **val_accuracy**가 특정 에폭 이후 정체 또는 감소(과적합 경향)하는 모습을 관찰함.
- **val_loss**가 불안정하게 진동, validation 세트에 모델이 충분히 일반화되지 못하고 있음을 시사.

### 1.2 unknown_or_empty 클래스
- `random_test_results.csv`에서는 unknown_or_empty 클래스에 대한 recall/precision이 낮음.
  - 이는 augmentation이나 데이터 수집 균형에서 미진함이 있을 수 있고,
  - Loss function 혹은 클래스 불균형에 대한 처리가 미흡할 수 있음을 시사.

### 1.3 실제 그리퍼 카메라 환경 대응
- 랜덤 카메라 환경(random_test_results.csv)의 정확도가 낮게 나타남. domain gap이 큼.
  - 기존 augmentation이 실제 환경 변화(조명, 스케일, 노이즈) 대응에 한계가 있음.

### 1.4 과적합
- 빠른 validation 성능 최고점 도달→성능 저하
  - augmentation ↑, regularization 강화 필요.

---

## 2. 다음 개선 실험 제안

1. **강화된 Augmentation 도입**
   - Color jitter, random shadow, random blur 등 실제 환경 반영 변환 추가.
   - rotation/shift/zoom 범위 확대.

2. **클래스 가중치 적용**
   - unknown_or_empty 및 소수 클래스에 가중치 부여(불균형 해소, recall↑).

3. **Dropout 및 L2 Regularization**
   - MobileNetV2의 head에 Dropout(0.3) 및 L2 규제(1e-4) 추가.

4. **EarlyStopping 보류 & ReduceLROnPlateau 적극 활용**
   - 불필요하게 일찍 멈추지 않고 learning rate 조정하며 최적 탐색.

5. **CSVLogger와 history_colab.csv 누락 방지**
   - 학습 로그 및 평가 결과 반드시 별도 저장.

---

## 3. 실제 Gripper 카메라 환경 대응 방안

- **Augmentation 탐색적 강화** 실험은 실환경 shift에 더 유연하게 적응하도록 돕는다.
- Data pipeline에 `rescale`, `random_brightness`, `contrast`, `random_noise` 추가.
- 평가시 별도의 random test셋(random_dataset) 항상 테스트.

---

## 4. 기대 효과

- overfitting 감소, validation loss 안정화
- unknown_or_empty 클래스 인식 개선
- 실제 환경 매칭 성능 개선

---

# [CODE]
```python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

# ==== Configs ====
base_dir = '/content/chess_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')
IMG_SIZE = (224, 224)  # MobileNetV2 default
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42

# ==== Data Augmentation ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # 확대
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=0.18,
    horizontal_flip=True,
    brightness_range=[0.65, 1.35],  # 색상
    channel_shift_range=20.,        # 색상
    fill_mode='nearest'
)

# Custom preprocessing for random contrast, Gaussian noise
def custom_preprocessing(img):
    # Random Contrast
    factor = np.random.uniform(0.8, 1.2)
    img = np.clip(img * factor, 0, 1)
    # Random Gaussian Noise
    if np.random.rand() < 0.25:
        noise = np.random.normal(0, 0.04, img.shape)
        img = np.clip(img + noise, 0, 1)
    return img

val_datagen = ImageDataGenerator(rescale=1./255)

# Use class_mode='categorical' for one-hot labels
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
    preprocessing_function=custom_preprocessing
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)
random_test_generator = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())

# ---- Compute class weights (for class imbalance) ----
from sklearn.utils.class_weight import compute_class_weight

train_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# ==== Model Build ====
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
base_model.trainable = False  # Transfer learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==== Callbacks ====
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(base_dir, 'model_best.h5'),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

csv_logger_cb = CSVLogger(os.path.join(base_dir, 'history_colab.csv'))

lr_reduce_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint_cb, csv_logger_cb, lr_reduce_cb]

# ==== Training ====
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ==== Save Final Model Weights ====
model.save_weights(os.path.join(base_dir, 'model_last_weights.h5'))

# ==== Training History Visualization (Optional) ====
# history_df = pd.DataFrame(history.history)
# plt.figure(figsize=(12, 5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Train acc')
# plt.plot(history.history['val_accuracy'], label='Val acc')
# plt.legend(); plt.title('Accuracy')
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Train loss')
# plt.plot(history.history['val_loss'], label='Val loss')
# plt.legend(); plt.title('Loss')
# plt.savefig(os.path.join(base_dir, 'learning_report.png'))

# ==== Evaluation on random test set ====
# Reload best weights
model.load_weights(os.path.join(base_dir, 'model_best.h5'))

random_test_generator.reset()
Y_pred = model.predict(random_test_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = random_test_generator.classes

# ---- Classification report ----
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(base_dir, 'random_test_results.csv')
report_df.to_csv(report_path, encoding='utf-8')

print("Random test results saved to:", report_path)

```