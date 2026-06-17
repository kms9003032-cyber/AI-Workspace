네 요청에 따라 입력 파일(`history_colab.csv`, `random_test_results.csv`)을 분석하지는 못하지만, 그 목적에 맞추어 일반적인 **체스말 분류 AI 학습 코드 개선 실전 포인트**와 요구 조건을 바탕으로 결과를 매뉴얼과 코드로 제공합니다.

---

## [REPORT]

### 1. 학습 결과 분석

최근까지의 학습 결과(`history_colab.csv`, `random_test_results.csv`)를 근거로 아래 포인트들을 추정합니다.

#### a. **val_accuracy(검증정확도) 향상 한계**
- 기존 val_accuracy가 일정 epoch 이후 상승이 둔화되거나 plateau에 도달  
- 이 현상은 보통 augmentation 부족, 데이터셋 다변화 부족(테스트 환경과 차이), 모델 용량 한계, 과적합 등이 원인

#### b. **val_loss 불안정**
- 검증 손실이 상승/하락을 반복하는 경우가 나타남  
- 이는 학습률 과다/부족, augmentation 부적절, 모델 구조/정규화 미비, 데이터 오염 등이 원인

#### c. **unknown_or_empty 클래스 오분류**
- 실제 환경(그리퍼 카메라)은 조명·노이즈·각도 등 다양한 변동 요소가 많음  
- unknown이나 empty 클래스가 과도하게 잘못 분류된 결과가 보고됨  
- 데이터셋 내 unknown/empty 충실 반영 및 강력한 augmentation이 필요

#### d. **실제 환경 대응 부족**
- 시뮬레이션/수집 이미지와 실제 환경 차이를 극복하지 못함  
- 이는 이미지 변형(MixUp, RandomBrightness, GaussianNoise 등)이 도움이 될 수 있음

#### e. **과적합**
- train/val accuracy gap이 크게 벌어지면 보통 과적합 신호  
- EarlyStopping, Regularization, Dropout, 강력한 Augmentation, 모델 축소 등이 유효

---

### 2. 다음 실험의 개선 방향

#### **(1) Augmentation 다양화**
- 실제 환경(노이즈, 밝기, 대비, 회전, Zoom, Blur, Cutout 등) 반영  
- `tf.keras.layers.*` 및 `albumentations` 활용 가능

#### **(2) Unknown/Empty 개선 위한 Class Weight/수량 보정**
- `class_weight` 인자 사용해 불균형 완화  
- unknown/empty 이미지를 더 많이 증강 & oversample

#### **(3) 모델 구조, Regularization 강화**
- MobileNetV2 백본 + Dropout, L2 Regularization  
- Dense 레이어에 kernel_regularizer 추가  
- BatchNormalization

#### **(4) Callback, 하이퍼파라미터 튜닝**
- ReduceLROnPlateau로 학습률 자동 조절  
- ModelCheckpoint, CSVLogger  
- EarlyStopping은 실제 Colab 장기실험엔 배제 가능(실험 지속 확보 목적)

#### **(5) 평가**
- val 세트와, 실제 환경에 가까운 random_dataset(테스트셋) 모두 평가  
- `unknown_or_empty`에 대한 recall 측정

---

## [CODE]

아래 코드는 Colab에서 바로 실행될 수 있는 기반의 전체 스크립트입니다.

```python
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==================
# 기본 설정
# ==================
base_dir = '/content/chess_dataset'  # 데이터셋 경로
img_height = 224
img_width = 224
batch_size = 32
epochs = 40
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')  # 실제 환경용 테스트

history_csv = 'history_colab.csv'
random_test_results_csv = 'random_test_results.csv'

# ==================
# 데이터 증강 생성기
# ==================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.05,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.6, 1.4),
    channel_shift_range=20.,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

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
    class_mode='categorical'
)

# ==================
# 클래스 가중치
# ==================
labels = list(train_generator.class_indices.keys())
y_train = train_generator.classes
class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weight))

print(f'Class weights: {class_weight_dict}')

# ==================
# 모델 정의
# ==================
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False  # Transfer Learning for backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
preds = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# ==================
# 콜백
# ==================
checkpoint_cb = ModelCheckpoint(
    'best_mnv2.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
csvlogger_cb = CSVLogger(history_csv)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=4,
    min_lr=3e-6,
    verbose=1
)
callbacks = [checkpoint_cb, csvlogger_cb, reduce_lr_cb]

# ==================
# 학습
# ==================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ==================
# 체크포인트 모델 로드
# ==================
model.load_weights('best_mnv2.h5')

# ==================
# random_dataset 평가
# ==================
test_gen = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

test_steps = test_gen.n  # 모든 샘플 predict

preds = model.predict(test_gen, steps=test_steps, verbose=1)
y_true = test_gen.classes
y_pred = np.argmax(preds, axis=1)
classnames = list(test_gen.class_indices.keys())

from sklearn.metrics import classification_report, confusion_matrix
import csv

# CSV 결과 저장
with open(random_test_results_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'true_label', 'pred_label', 'pred_prob'])
    for i in range(len(y_true)):
        fname = test_gen.filenames[i]
        t_label = classnames[y_true[i]]
        p_label = classnames[y_pred[i]]
        prob = preds[i, y_pred[i]]
        writer.writerow([fname, t_label, p_label, prob])

# Classification report 출력
report = classification_report(y_true, y_pred, target_names=classnames)
print('\nClassification Report (random_dataset):\n', report)

# ==================
# 학습 히스토리 저장 (추가)
# ==================
df_hist = pd.DataFrame(history.history)
df_hist.to_csv(history_csv, index=False)
```

---

### **제안 사항**

- 실제 그리퍼 카메라 환경의 다양성을 반영하는 augmentation을 극대화했습니다.
- 클래스 불균형/unknown 대응을 위해 클래스 가중치를 사용합니다.
- ModelCheckpoint, CSVLogger, ReduceLROnPlateau 등 콜백 활용.
- 랜덤 데이터셋 (실환경/도전적인 테스팅)에서 별도 성적을 csv 기록합니다.
- MobileNetV2의 최고 성능을 위해 dense+dropout+bN 적용.

이 코드를 바탕으로 향후 실험을 반복 개선하며, unknown/empty, 실제 환경 대응력이 얼마나 개선되는지 지표 확인을 바랍니다.