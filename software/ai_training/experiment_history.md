아래 분석과 코드는 아래와 같은 체스말 분류 문제에 효과적으로 대응하기 위해 작성되었습니다.

---

# [REPORT]

## 1. 데이터 및 현재 결과 분석

### **history_colab.csv**
- **val_accuracy**: 훈련 정확도보다 상대적으로 낮거나 변동폭이 심하다면 overfitting 징후.
- **val_loss**: 불안정한 경우, 모델의 일반화 성능이 낮음. 학습률, 모델 복잡도, regularization 등 개선 필요.
- **unknown_or_empty**: unknown, empty 클래스를 혼동/미인식하는 경우가 많다. 실제 환경(조명, 각도, 배경)에서 분포가 다를 수 있음.

### **random_test_results.csv**
- 랜덤 샘플에서 성능 하락이 관찰된다면, 실제 카메라 환경 적응력이 부족함을 의미.  
- 특정 클래스의 recall/precision이 낮다면 데이터 불균형, overfitting 원인일 수 있음.

---

## 2. 다음 실험 설계 이유

### 대표 개선 전략

- **val_accuracy 향상/val_loss 안정화**:  
  - `ReduceLROnPlateau`로 validation 성능 정체 시 학습률 자동 감소.
  - data augmentation 다양화(색 변화, noise 등) 및 dropout, batch normalization 추가.
- **unknown_or_empty 개선**:  
  - 클래스 균형을 위한 oversampling이나 class_weight 적용.
  - augmentation을 특히 empty, unknown 쪽에 강화.
- **실제 카메라 환경 대응**:  
  - 랜덤한 augmentation(조명, blur, 변형 등) 추강.
  - random_test_results로 직접 평가/저장.  
- **과적합 방지**:  
  - 모델 단순화, 과한 에폭 제한, dropout, early stopping(모델 checkpoint), aggressive aug 등.

---

## 3. 실험 주요 변경점

- **데이터 파이프라인**: `ImageDataGenerator`에 밝기, 노이즈, 회전, shift 등 강한 augmentation. empty/unknown 데이터가 부족하면, train/val에서 각각 비율 유지.
- **모델**: MobileNetV2 (pretrained, fine-tuning layer 조절)
- **callback**: `ModelCheckpoint`, `CSVLogger`, `ReduceLROnPlateau`
- **과적합 방지**: dropout 추가, EPOCH 적정 제한(30~50).
- **random_test_results**: test 데이터셋 사용해서 별도 평가.

---

# [CODE]
```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# Set base_dir and dataset paths
base_dir = '/content/chess_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 40

# Data Augmentation for real camera robustness
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=(0.6, 1.4),
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,  # 추가로 실제 환경 적용
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Train/Validation Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# # Identify class imbalance
# Consider applying class_weight if unknown/empty is underrepresented
from collections import Counter
counter = Counter(train_generator.classes)
max_count = max(counter.values())
class_weight = {i: max_count/count for i, count in counter.items()}

# Build MobileNetV2 Model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # For transfer learning, Fine-tune later

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint_cb = ModelCheckpoint(
    'best_model.h5', monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max')
csv_logger_cb = CSVLogger('history_colab.csv')
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    verbose=2, min_lr=1e-5)

callbacks = [checkpoint_cb, csv_logger_cb, reduce_lr_cb]

# Train the model (first stage: head only)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks
)

# Optionally fine-tune (second stage)
base_model.trainable = True
optimizer_finetune = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer_finetune,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=8,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks
)

# Save final history to CSV (append/merge if secondary phase)
def merge_histories(history_list):
    dfs = []
    for h in history_list:
        df = pd.DataFrame(h.history)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

full_history = merge_histories([history, history_fine])
full_history.to_csv('history_colab.csv', index=False)

# Evaluate on random_dataset (_test_ in actual camera environment)
random_test_gen = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
eval_result = model.evaluate(random_test_gen, verbose=1)
pred_probs = model.predict(random_test_gen, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = random_test_gen.classes

# Calculate per-class accuracy, confusion, unknown_or_empty 개선 상황 분석
from sklearn.metrics import classification_report, confusion_matrix

target_names = list(random_test_gen.class_indices.keys())
report = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Save detailed test results
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('random_test_results.csv')
# For extra insight (optional):
# pd.DataFrame(conf_matrix, columns=target_names, index=target_names).to_csv('random_test_confusion.csv')

print("Evaluation Results (accuracy, loss):", eval_result)
print("Classification Report saved to random_test_results.csv")
```

---

**설명:**  
- **강화된 augmentation**으로 일반화와 unknown/empty robustness 개선을 겨냥
- **class_weight 적용**으로 불균형(특히 empty/unknown)에 대응
- **ReduceLROnPlateau**와 **dropout**, **batch norm** 등 안정화 및 과적합 방지 적용
- **random_dataset** 평가 및 **csv 저장**  
- **history_colab.csv**: 전체 학습 내역 저장  
- **random_test_results.csv**: per-class 리포트, unknown/empty 개선 직접 확인

필요시, fine-tune 단계나 augmentation 강도, layer unfreeze 비율 등 추가 조정하며 실험 반복을 권장합니다.