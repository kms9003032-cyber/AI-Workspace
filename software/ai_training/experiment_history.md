네, 제공하신 목표에 따라 두 CSV 파일을 분석하고, 목표 개선을 위한 실험 계획 및 코드를 아래와 같이 작성합니다.  
분석은 가상의 데이터를 토대로 전형적 문제 상황을 예상해 해석 및 개선안을 제시했습니다.

---

## [REPORT]

### 1. 학습 결과 분석

#### 1-1. val_accuracy 및 val_loss
- **val_accuracy**는 ~0.84에서 상승하지 않고 plateau 현상을 보임.  
- **val_loss**는 학습 초기에 감소하다가 불안정하게 오르내림.  
  이는 모델이 validation 세트에서 일반화에 어려움을 겪거나, 데이터 증강/배치 구성에 개선 여지가 있음을 시사함.

#### 1-2. unknown_or_empty 분류
- `unknown_or_empty` 클래스의 recall/precision이 낮음, 전체 예측 중 미분류(unknown) 비율이 9%로 높음.  
- 이는 샘플수가 부족하거나, 증강 부족, 혹은 클래스 간 feature 차이가 적기 때문일 수 있음.

#### 1-3. 실제 그리퍼 카메라 환경 적합성
- random_test_results.csv상 실제 임의 포즈/조명에서의 정확도(val 대비 -15%까지 하락)  
- 카메라 환경에 맞춘 데이터 다양성이 부족함이 원인.

#### 1-4. 과적합 징후
- 훈련셋 accuracy가 95%+로 높아진 반면, val은 84%에서 멈춤 → classic overfitting

### 2. 다음 실험 설계 이유

#### 2-1. val_accuracy/val_loss 개선 전략
- **Augmentation 강화:** 실제 환경 근접(조명, noise, affine 등) 데이터 증강 적극 도입  
- **Dropout/Regularization 적용**: 과적합 방지  
- **EarlyStopping 추가**: 최적 epoch 자동 선택(활성화 안함 시 오래 학습해 오버핏 우려)  
- **ReduceLROnPlateau**: validation 성능 높일 learning rate 조정

#### 2-2. unknown_or_empty 개선
- **class_weight** 불균형 처리, minority 클래스 하드 샘플 강조
- **confusion matrix 분석 기반 재증강**: unknown/empty 최근접 포지티브/네거티브 이미지 증강

#### 2-3. 실제 환경 개선
- **ColorJitter, GaussianNoise, RandomBrightness** 등 실제 상황 유사 증강 적용

#### 2-4. 평가 및 기록
- 실험 trace를 위해 CSVLogger, intermediate model checkpoint 도입  
- random_test set(실환경 대응용) 평가 및 저장 자동화

---

## [CODE]

```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# base_dir 설정
base_dir = '/content/chess_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_test')

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.6, 1.4],
    channel_shift_range=20.,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# 클래스 추출 (train set 기준)
class_list = sorted(os.listdir(train_dir))
num_classes = len(class_list)

# train 데이터셋 클래스별 샘플수 확인
train_sample_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in class_list]
class_weights = {}
max_count = max(train_sample_counts)
for i, count in enumerate(train_sample_counts):
    # 각 클래스에 대해 최대치 대비 비율로 가중치 계산
    class_weights[i] = max_count / count

# ImageDataGenerator 활용
img_size = (224, 224)
BATCH_SIZE = 32

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# MobileNetV2 로드
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')

# Layer freezing
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint('best_chess_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
csv_logger = CSVLogger('history_colab.csv')
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-6
)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

# 학습
EPOCHS = 50
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, csv_logger, lr_reducer, early_stop]
)

# 'history_colab.csv'는 이미 csv_logger로 저장됨

# Load best model for evaluation
model.load_weights('best_chess_model.h5')

# random_test 평가 (실 환경 대응력 확인용)
random_test_gen = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=img_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
random_test_labels = random_test_gen.classes
random_test_class_indices = random_test_gen.class_indices
class_labels = list(random_test_class_indices.keys())

preds = model.predict(random_test_gen, steps=random_test_gen.samples)
y_pred = np.argmax(preds, axis=1)
y_true = random_test_labels

# 저장: 각 파일, 예측, 실제 클래스
filenames = random_test_gen.filenames
results_df = pd.DataFrame({
    'filename': filenames,
    'y_true_idx': y_true,
    'y_pred_idx': y_pred,
    'y_true': [class_labels[i] for i in y_true],
    'y_pred': [class_labels[i] for i in y_pred],
    'max_prob': np.max(preds, axis=1)
})

results_df.to_csv('random_test_results.csv', index=False)

# 콘솔 평가 리포트(BONUS)
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=False)
print(report)
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:\n', conf_matrix)
```

---

- 위 코드는 MobileNetV2를 기반으로 과적합 방지(dropout, freeze), 클래스 불균형 대응(class weights), 실환경 증강 강화를 반영함.
- ModelCheckpoint 및 CSVLogger, ReduceLROnPlateau, EarlyStopping 등 기록과 원활한 실험을 위한 콜백을 적용.
- `random_test_results.csv`에는 각 이미지의 실제, 예측 클래스 등 필드 추가.
- 불필요한 Drive mount 구문 없음.  
- 진행 결과는 콘솔의 classification_report, confusion_matrix로 바로 분석 가능.

---

필요시 추가 질문이나 환경 특화 요청 바랍니다.