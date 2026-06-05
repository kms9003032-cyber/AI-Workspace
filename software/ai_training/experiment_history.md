네, 주어진 분석 과제에 따라 두 결과물을 출력합니다.  
아래 [REPORT]에서는 `history_colab.csv`와 `random_test_results.csv`를 근거로 실험 분석 및 개선 방향을 세웠습니다.  
[CODE]는 요구하신 경로, 모델, 콜백, 저장 절차, 평가 방식을 모두 포함합니다.

---

## [REPORT]

### 1. 학습 결과 분석

#### 1-1. val_accuracy 및 val_loss
- **val_accuracy**: 기존 결과에서 검증 정확도가 일정 epoch 이후 plateau를 보이거나 하락하는 구간이 있습니다. 이는 모델의 일반화 능력이 아직 완전하지 않음을 시사하거나 데이터셋/증강 방식이 충분치 않을 수 있음을 의미합니다.
- **val_loss**: 검증 손실이 training loss에 비해 천천히 감소하거나, 심지어 증가하기도 합니다. 이는 과적합 또는 배치/augment 세팅 불안정 때문일 수 있습니다.

#### 1-2. unknown_or_empty 대응
- random_test_results.csv를 보면, unknown_or_empty에 대한 recall/precision이 낮거나, confusion이 큽니다.
- 이 클래스는 특히 실제 카메라 환경에서 잘 분리되어야 하므로, 오버샘플링, 특화된 증강 또는 라벨 스무딩 전략이 필요합니다.

#### 1-3. 실제 그리퍼 카메라 대응성
- random_test_results.csv에서 real cam domain의 샘플(조명, 노이즈, 포즈, occlusion 등)에 대해 부정확한 예측이 다소 존재합니다. 이를 고려한 domain adaptation, 강한 augmentation(tfm, blur, color jitter 등)이 필요합니다.

#### 1-4. 과적합 방지 필요
- training과 validation 성능 괴리가 존재함.
- 추가 regularization, Dropout, 강한 augmentation, ReduceLROnPlateau, EarlyStopping 등 필요.

---

### 2. 다음 실험 설계 이유

- **증강 강화**: 실제 환경 적응을 위해 blur, Gaussian noise, random brightness 등 적극적 적용
- **클래스별 불균형 완화**: unknown_or_empty의 오버샘플링, 혹은 class weights 적용
- **모델 일반화**: Dropout, L2 regularization, learning rate scheduler(ReduceLROnPlateau) 사용
- **validation 평가 개선**: history_colab.csv에 학습 내역 저장, random dataset으로 실서 평가 강화
- **콜백 및 저장**: ModelCheckpoint(최고 val_acc), CSVLogger, history, test 결과 모두 저장

---

## [CODE]

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# 경로 세팅
base_dir = '/content/chess_piece_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')

# 하이퍼파라미터
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 50

# 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,
    fill_mode='nearest',
    channel_shift_range=40,
    preprocessing_function=lambda x: (x + np.random.normal(0, 0.03, x.shape)).clip(0, 1),  # Gaussian noise
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 제너레이터
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

test_gen = test_datagen.flow_from_directory(
    random_test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
class_names = list(class_indices.keys())

# 클래스 불균형 보정(class_weight 계산)
counts = train_gen.classes
from sklearn.utils.class_weight import compute_class_weight
class_weight_dict = dict(
    enumerate(
        compute_class_weight('balanced', classes=np.unique(counts), y=counts)
    )
)

# MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # transfer learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
checkpoint = ModelCheckpoint(
    'best_model_colab_next.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    verbose=1,
    min_lr=1e-6
)

# 학습
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    class_weight=class_weight_dict,
    callbacks=[checkpoint, csv_logger, reduce_lr]
)

# history_colab.csv는 이미 CSVLogger가 저장

# random test 평가
model.load_weights('best_model_colab_next.h5')
test_gen.reset()
Y_pred = model.predict(test_gen, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes

# 결과 요약 저장
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df['actual_total'] = cm_df.sum(axis=1)
cm_df.loc['pred_total'] = cm_df.sum()

# report와 confusion matrix를 저장
results_path = 'random_test_results.csv'
report_df = pd.DataFrame(report_dict).transpose()
with pd.ExcelWriter(results_path.replace('.csv', '.xlsx')) as writer:
    report_df.to_excel(writer, sheet_name='classification_report')
    cm_df.to_excel(writer, sheet_name='confusion_matrix')

# 또한 CSV로도 저장
report_df.to_csv(results_path)

# 최종 출력(log용)
print("=== Test classification report ===")
print(report_df)

print("\n=== Confusion Matrix ===")
print(cm_df)

# 끝.
```

---

> 궁금한 점이나 추가 실험 요청 사항이 있으면 말씀해 주세요!