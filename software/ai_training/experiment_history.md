좋은 질문 감사합니다! 아래의 흐름을 따라 분석과 다음 실험 코드를 제공합니다.

---

## [REPORT]

### 1. 학습 결과 분석

#### val_accuracy & val_loss
- **val_accuracy**는 일정 epoch 후 정체 또는 하락하는 경향이 있습니다.
- **val_loss**는 후반부 불안정하게 증가(진동)하거나, 감소 추세가 약한 경향을 보입니다.
- 이는 현재 모델 구조 혹은 augmentation 및 regularization이 불충분하거나 learning rate가 고정되어 있기 때문일 수 있습니다.

#### unknown_or_empty 개선 필요
- random_test_results.csv 분석 결과, unknown_or_empty 클래스(또는 미분류/오인식)에서 객체 오분류 비율이 높습니다.
- 실제 그리퍼 카메라 환경에서 노이즈 및 조명 변화, 빈 영역 등 핸들링 부족으로 인해 false positive/negative가 발생할 수 있습니다.

#### 과적합 문제
- 훈련 효율을 높여도 val set에 대한 성능이 plateau에 머무르는 현상이 관측됩니다.
- 이는 augmentation 부족 혹은 모델 capacity가 과하거나 regularization이 부족하기 때문일 수 있습니다.

---

### 2. 다음 실험 설계 및 이유

1. **모델 개선**
   - MobileNetV2를 계속 사용하되, dropout 및 GlobalAveragePooling 추가로 과적합 방지
   - BatchNormalization 추가

2. **Data Augmentation**
   - RandomBrightness, RandomContrast, RandomZoom, RandomTranslation 등 실제 환경을 모사하는 augmentation 강화
   - unknown_or_empty 클래스가 포함된 negative sample oversampling

3. **Learning Rate & Callback**
   - ReduceLROnPlateau 도입해 val_loss가 줄지 않으면 learning rate를 감소시킴
   - EarlyStopping(미포함 시)이나 정규 checkpoint으로 과적합 방지

4. **실제 환경 대응**
   - 이미지 해상도 소폭 조정(224x224 등)
   - Augmentation을 실제 gripper 카메라 환경에 맞게 설계
   - [optional] background noise나 blur, occlusion 포함

5. **평가**
   - unknown_or_empty 및 class별 성능을 random_test_results.csv로 상세 저장

---

## [CODE]

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# --- 경로 설정
base_dir = '/content/drive/MyDrive/chess_piece_classifier'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')  # 평가용

# --- 하이퍼파라미터
batch_size = 32
img_height = 224
img_width = 224
epochs = 60
initial_lr = 1e-3

# --- 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=0.15,
    brightness_range=(0.7, 1.3),
    channel_shift_range=15.0,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# --- 제너레이터
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

# --- 모델 정의
base_model = MobileNetV2(include_top=False,
                         input_shape=(img_height, img_width, 3),
                         weights='imagenet')
base_model.trainable = False  # Fine-tuning은 이후 고려

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

optimizer = Adam(learning_rate=initial_lr)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 콜백
checkpoint_path = os.path.join(base_dir, 'best_model.h5')
csv_logger_path = os.path.join(base_dir, 'history_colab.csv')
best_model_ckpt = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1,
                                  save_best_only=True, save_weights_only=False, mode='max')
csv_logger = CSVLogger(csv_logger_path, append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=2, min_lr=1e-6)

callbacks = [best_model_ckpt, csv_logger, reduce_lr]

# --- 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=callbacks
)

# --- History 추가 저장 (데이터프레임으로 csv 다시 저장)
history_df = pd.DataFrame(history.history)
history_df.to_csv(csv_logger_path, index=False)

# --- 랜덤 데이터셋 평가
random_test_datagen = ImageDataGenerator(rescale=1./255)
random_test_generator = random_test_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
random_test_labels = random_test_generator.classes
random_test_filenames = random_test_generator.filenames
y_pred_probs = model.predict(random_test_generator, steps=len(random_test_generator), verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

results_dict = {
    'filename': random_test_filenames,
    'true_class': [inv_class_indices[c] for c in random_test_labels],
    'pred_class': [inv_class_indices[c] for c in y_pred],
    'is_correct': (random_test_labels == y_pred)
}
for idx in range(num_classes):
    class_name = inv_class_indices[idx]
    results_dict[f'prob_{class_name}'] = y_pred_probs[:, idx]

random_results_df = pd.DataFrame(results_dict)
random_results_csv_path = os.path.join(base_dir, 'random_test_results.csv')
random_results_df.to_csv(random_results_csv_path, index=False)

# --- class별 unknown_or_empty 비율 등 통계 콘솔 출력
unknown_class_idx = None
for k, v in class_indices.items():
    if 'unknown' in k or 'empty' in k:
        unknown_class_idx = v
        break

if unknown_class_idx is not None:
    unknown_or_empty_pred_count = (y_pred == unknown_class_idx).sum()
    print('Predicted unknown_or_empty:', unknown_or_empty_pred_count)
    print('True unknown_or_empty samples:', (random_test_labels == unknown_class_idx).sum())
    print('Unknown/Empty 정확도:', (random_test_labels[(y_pred==unknown_class_idx)] == unknown_class_idx).sum())

print(f"전체 test acc: {np.mean(random_test_labels==y_pred):.4f}")

# --- 코드 끝
```

---

#### 실행 전 필수 체크
- base_dir, train_dir, val_dir, random_test_dir의 경로가 환경과 일치하는지 확인
- Colab의 런타임을 리셋한 경우, 필요한 라이브러리(tf, pandas, numpy)가 런타임에 설치되어 있어야 함
- `/content/drive/...` 경로가 마운트 및 접근 가능한 상태여야 함

---

원하시는 추가 조정 사항이 있으면 언제든 말씀해 주세요!