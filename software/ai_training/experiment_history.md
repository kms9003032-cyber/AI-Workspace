분석 및 개선 내용은 아래와 같습니다.

---

# [REPORT]

## 1. 학습 결과 분석

### history_colab.csv (학습 로그)
- **val_accuracy**: 에폭 후기에 오르지 않거나 정체되는 경향이 있다.
- **val_loss**: 종종 불안정하게 튀는 값. 과적합 신호 있음 (train/val gap).
- **unknown_or_empty**: 해당 클래스의 recall, precision 모두 낮음. 실제 환경서 비인식 확률 높음.
- **과적합**: 후반부 val_loss 상승, val_accuracy 변화 미미 → 과적합 의심.

### random_test_results.csv (실제 random set 평가)
- **레이블 오염** 가능성, 여러 클래스에서 unknown_or_empty 오탐지.
- **val/test accuracy 차이**: 실제 환경에 대한 일반화 부족.
- **특정 클래스 저성능**: 그리퍼 or 그림자 섞인 이미지서 성능저하.

---

## 2. 실험 방향 및 개선 이유

- **Data Augmentation**: 현실 잡음(그리퍼, 조명, 움직임 등) 반영 위해 Aggressive하게 적용 (Rotation, Shear, Cutout, RandomBrightness 등)
- **Pretrained MobileNetV2**: Transfer Learning으로 과적합 방지 및 일반화.
- **Class weights**: unknown_or_empty 등 난이도 높은 클래스에 가중치 부여.
- **ReduceLROnPlateau**: val_loss unstable할 때 학습 안정 및 escape local minima.
- **EarlyStopping 고려**: validation loss 오래 개선 없을 때 종료 → colab resource 절약 및 overfitting 방지.
- **ModelCheckpoint**: best val_loss 기준 저장.
- **BatchNormalization & Dropout**: 추가해 과적합 방지.
- **random_dataset 평가**: production 환경 반영.
- **CSVLogger, history 저장**: reproducibility.

### 실험 요약
- Aggressive augmentation으로 실제 환경/노이즈 견고성 향상.
- 과적합 억제 (Dropout, batchnorm, class weight, ReduceLROnPlateau, 작은 batch size)
- 평가(unknown, random set) 코드 일원화.

---

# [CODE]


```python
import os
import numpy as np
import pandas as pd
from glob import glob

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

# ================
# Parameters
# ================
base_dir = '/content/chess_dataset'  # 반드시 절대경로로 수정 필요
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')  # 실제 환경 대응 데이터

BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
EPOCHS = 50

# ==============================
# Data Augmentation
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    channel_shift_range=20,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

# ====================
# Class Weights
# ====================
classes = list(train_generator.class_indices.keys())
class_counts = [len(glob(os.path.join(train_dir, c, '*'))) for c in classes]
total = np.sum(class_counts)
class_weights = {i: total/(len(classes)*n) for i, n in enumerate(class_counts)}

# unknown_or_empty에 weight 1.5~2.0 부여 (만약 해당 클래스를 포함한다면)
unknown_idx = [i for i, cls in enumerate(classes) if ('unknown' in cls) or ('empty' in cls)]
for i in unknown_idx:
    class_weights[i] = class_weights[i] * 2.0

# ====================
# Model
# ====================
base_model = MobileNetV2(input_shape=TARGET_SIZE+(3,), include_top=False, weights='imagenet')
base_model.trainable = False  # transfer learning, fine-tuning은 이후 단계

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
outputs = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====================
# Callbacks
# ====================
checkpoint_cb = ModelCheckpoint(
    'best_model_colab.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

csv_logger = CSVLogger('history_colab.csv')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint_cb, csv_logger, reduce_lr, early_stopping]

# ====================
# Train
# ====================
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
)

# ============================
# Fine-tuning (optional)
# ============================
# Optionally unfreeze deeper layers for 5-10 epochs if overfitting 안 할 때만
# Uncomment if want further improvement
# base_model.trainable = True
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# history_finetune = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=5,
#     validation_data=val_generator,
#     validation_steps=val_generator.samples // BATCH_SIZE,
#     callbacks=callbacks,
#     class_weight=class_weights
# )

# ============================
# Save history as CSV (again, ensure last epoch saved)
# ============================
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_colab.csv', index=False)

# ============================
# Evaluation: random_dataset
# ============================
def evaluate_on_directory(model, directory, class_indices, batch_size=32):
    # Build generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )
    gen = datagen.flow_from_directory(
        directory,
        target_size=TARGET_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    Y_pred = model.predict(gen, steps=np.ceil(gen.samples / batch_size))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = gen.classes
    
    # class name 매핑
    idx2class = {v:k for k, v in class_indices.items()}
    # 전체 accuracy
    acc = np.mean(y_pred == y_true)
    # 각 클래스별 정밀도/재현율
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=[idx2class[i] for i in range(len(idx2class))], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    # 주요 값만 요약 DataFrame으로 저장
    df_summary = pd.DataFrame({
        'class': [idx2class[i] for i in range(len(idx2class))],
        'precision': [report[idx2class[i]]['precision'] for i in range(len(idx2class))],
        'recall': [report[idx2class[i]]['recall'] for i in range(len(idx2class))],
        'support': [report[idx2class[i]]['support'] for i in range(len(idx2class))]
    })
    # 전체 accuracy 추가
    df_summary.loc[len(df_summary)] = ['overall', acc, acc, sum(df_summary['support'])]
    return df_summary, cm

# 베스트 모델 로드 (혹시 fine-tune 후라면 reload하는게 안전)
best_model = tf.keras.models.load_model('best_model_colab.h5')

random_test_results, conf_mat = evaluate_on_directory(best_model, random_test_dir, train_generator.class_indices, batch_size=BATCH_SIZE)
random_test_results.to_csv('random_test_results.csv', index=False)

# 전체 confusion matrix도 저장 (옵션)
cm_df = pd.DataFrame(conf_mat, index=classes, columns=classes)
cm_df.to_csv('random_test_confusion_matrix.csv')

print("Train/validation history saved to history_colab.csv")
print("Random set test results saved to random_test_results.csv")

```
---

**노트:**
- 기본 `base_dir`, `train_dir`, `val_dir`, `random_test_dir`는 환경에 맞게 수정 필요
- 추가적으로 EarlyStopping은 실험자 재량으로 활용가능, EPOCHS도 리소스에 맞게 조절
- 코드가 한 셀에서 Colab에 동작하도록 설계되어 있음

---

**혹시 필요시 augmentation strength, fine-tuning depth, dropout rate 등 실험적으로 튜닝할 것!**