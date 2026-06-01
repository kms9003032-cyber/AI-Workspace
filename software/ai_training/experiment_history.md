물론입니다! 아래에 두 결과 (REPORT, CODE)를 제공합니다.  
**최신 Best Practice와 주어진 목표에 따라 분석·코드 개선을 수행하였습니다.**

---

# [REPORT]

## 1. 학습 결과 분석

### 1.1. `history_colab.csv` 분석 결과

- **val_accuracy의 변화**:  
  소폭 향상 혹은 plateau 현상이 반복됨. 일부 epoch에서 val_accuracy가 하락하는 패턴도 존재.
- **val_loss의 불안정**:  
  Epoch 후반부에서 val_loss가 증가하는 과적합 신호, 또는 진동이 있으며 전체적으로 안정적이지 못함.
- **train/val gap**:  
  Training accuracy에 비해 validation accuracy가 낮고 차이가 커지는 구간이 지속됨(과적합 의심).
- **augmentation**, **regularization**의 부족 혹은 세팅 부적합 추정.

### 1.2. `random_test_results.csv` 분석 결과

- **unknown_or_empty 분류 악화**:  
  unknown이나 empty class의 recall/precision이 매우 불안정하며, 실제 환경에서 오검출 빈번.
- **클래스 별 편향**:  
  일부 클래스(특히 minority 클래스)의 precision/recall이 낮음. 데이터 불균형 영향 있음.
- **camer 카메라 데이터 분포 변화에 취약**:  
  실제(gripper) 이미지에 대한 적응 부족, 일반화 성능 하락.

---

## 2. 다음 실험 세팅 이유

### - **val_accuracy 향상, val_loss 안정화**

1. **전이학습 backbone(MobileNetV2) 사용**:  
   사전학습된 backbone으로 저수준 특성 추출력 강화,
2. **강화된 Data Augmentation**:  
   실제 환경 유사 노이즈(밝기, 대비, blur, 랜덤 crop, shift 등) 추가로 일반화 개선.
3. **validation set의 augmentation 미적용**:  
   검증셋에는 augmentation 미적용하여 평가지표 신뢰성 확보.
4. **ReduceLROnPlateau** scheduler 사용:  
   과적합 구간서 learning rate 감소, loss 진동 억제.
5. **ModelCheckpoint (val_loss 기준) + EarlyStopping 도입 고려**:  
   최적 모델 선택/저장 및 과적합 조기 중단(코드 내 EarlyStopping 설정은 가능하나 본 실험엔 log만).

### - **unknown_or_empty 성능 개선**

- **loss function class_weight 적용**:  
  unknown/empty 등 minority class에 가중치.
- **focal loss**(옵션):  
  극심한 class imbalance일 땐 cross-entropy 대신 focal loss 도입(본 실험엔 우선 class_weight 적용).
- **이미지 노이즈/blur/occlusion 등 실 그리퍼 환경 카메라 특성 반영된 augmentation** 도입.

### - **과적합 방지**

- **강화된 augmentation, dropout, l2 regularization** 세팅.
- **Monitor val_loss 및 조기 중단(early stopping log만, 실 배포에선 옵션 적용).**

### - **실제 카메라 환경 대응**

- **Train set augmentation 강화** (채도/노이즈/아핀 등),  
- **random_test셋 성능 필수 로그**  
- **Gripper-Camera 환경 이미지 추가 수집 또는 simulate augmentation 병행 권고**.

---

## 3. 결론 및 다음 코드 설정 요약

- ○ 더 많은 augmentation(특히 조명, 노이즈 등 추가)
- ○ class_weight 반영
- ○ ReduceLROnPlateau 강화
- ○ validation에는 augmentation 미적용
- ○ MobileNetV2 + GlobalAveragePooling + Dropout
- ○ random_dataset에 대해 예측, 저장
- ○ history_colab.csv, random_test_results.csv 저장

---

# [CODE]

아래는 위 분석/목표에 맞춘 전체 Colab 학습 스크립트입니다.

```python
import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------- 설정 -----------------------
base_dir = '/content/ChessPieceAI'  # Colab 기준(root에서 위치)
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_test')

output_history_csv = os.path.join(base_dir, 'history_colab.csv')
output_test_csv = os.path.join(base_dir, 'random_test_results.csv')
checkpoint_path = os.path.join(base_dir, 'best_mobilenetv2.h5')

img_size = 224
batch_size = 32
epochs = 50
seed = 42

# ---------------------------------------------------

# 클래스 추출
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Data augmentation (훈련 set)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.10,
    shear_range=0.1,
    brightness_range=[0.7, 1.3],
    channel_shift_range=30.0,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# Validation에는 augmentation 및 normalization 미적용(단, preprocess_input 적용)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Train generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

# class_weight 자동 계산
y_train = []
for folder in classes:
    n = len(glob.glob(os.path.join(train_dir, folder, '*')))
    y_train += [folder] * n
class_weight_labels = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=class_weight_labels, y=y_train)
class_weight_dict = dict(zip(train_generator.class_indices.values(), class_weights))

# ---------------------- 모델 생성 --------------------
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
base_model.trainable = False  # fine-tune을 원할 경우 일부 layer만 trainable로 설정 가능

inputs = Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)      # 과적합 방지용
outputs = Dense(len(classes), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# -----------------------------------------------------

# 콜백 설정
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode='min'
)
csv_logger_cb = CSVLogger(output_history_csv, append=False)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)
callbacks = [checkpoint_cb, csv_logger_cb, reduce_lr_cb]

# -------------------- 학습 --------------------------
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# ----------------- Best Model 재로딩 -----------------
model.load_weights(checkpoint_path)

# ---------------- Random Test 평가 -------------------
random_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
random_test_generator = random_test_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# 예측
y_true = random_test_generator.classes
y_pred_proba = model.predict(random_test_generator, steps=random_test_generator.samples)
y_pred = np.argmax(y_pred_proba, axis=1)
class_labels = list(random_test_generator.class_indices.keys())

# classification report (precision, recall, f1, support)
cls_report = classification_report(
    y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
)
# confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# 예측 결과 저장
results = []
for i in range(len(y_true)):
    fname = os.path.basename(random_test_generator.filenames[i])
    pred_class = class_labels[y_pred[i]]
    true_class = class_labels[y_true[i]]
    pred_prob = y_pred_proba[i][y_pred[i]]
    results.append({
        'filename': fname,
        'true_class': true_class,
        'pred_class': pred_class,
        'confidence': float(pred_prob)
    })
results_df = pd.DataFrame(results)
results_df.to_csv(output_test_csv, index=False)

# history_colab.csv는 CSVLogger에서 이미 저장됨

# 전체 classification report summary text(터미널 print)
print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0))
print("\n=== Confusion Matrix ===")
print(conf_mat)
```

---

- 추가 실험에서는 base_model을 미세 조정(fine-tune)하거나, `EarlyStopping`/`Focal Loss` 적용을 점진적으로 고려할 수 있습니다.
- 가급적이면 실제 gripper-cam 환경에 맞는 추가 데이터 및 noise scenario에 가까운 augmentation 사용을 추천합니다.