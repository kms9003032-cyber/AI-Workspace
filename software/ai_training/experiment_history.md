네, 아래에 두 결과물을 차례로 제공합니다.

---

## [REPORT]
### 1. 학습 결과 분석 (history_colab.csv, random_test_results.csv)

#### (1) Val Accuracy/Val Loss 분석  
- **val_accuracy**: 일반적으로 train accuracy 대비 낮거나, 불안정한 증가 곡선을 보임. 이는 validation set 일반화에 어려움이 있고, 더 나은 일반화 전략이 필요함을 시사.  
- **val_loss**: 학습 초반 급격히 감소 후 plateau, 이후 training loss와의 간격이 벌어지는 양상이 나타남. 이는 과적합 소지가 있으며, 학습이 진행됨에 따라 validation loss가 다시 증가하는 구간도 있음(=early stopping 신호).  
- **불안정 구간**: augmentation 효과가 부족하거나, 하이퍼파라미터(learning rate 등) 최적화가 미흡.

#### (2) Unknown/Empty 데이터 예측 성능  
- **random_test_results.csv에서 unknown_or_empty 클래스의 정확도**가 다른 클래스에 비해 낮게 나타남.  
- 이는 unknown/empty 샘플에 대한 데이터 다양성 부족, 혹은 class imbalance, augmentation 미흡 등이 원인.

#### (3) 실제 카메라(그리퍼) 환경 대응력  
- random test set에서 성능 하락이 뚜렷함 → overfitting 혹은 train/test 도메인 차이 존재.  
- 실제 환경에서의 잡음, 변형, 조명 차이, 카메라 화질 및 왜곡 등 외적 요인에 대한 robustness 부족.

---

### 2. 다음 실험 방향 및 이유

1. **데이터 증강(Augmentation) 강화**  
   - 밝기, 명암, 채도 조정, Gaussian noise 추가, 원근/왜곡, random crop/zoom 등 현실 카메라 환경 유사 augmentation을 강화한다.  
   - 특히 unknown/empty 샘플의 augmentation을 적극적으로 적용하여 일반화 성능 향상.

2. **MobileNetV2 사전학습 기반 및 학습률 관리**  
   - MobileNetV2를 ImageNet으로 사전학습(pretrained) 후 fine-tuning한다.
   - ReduceLROnPlateau 콜백으로, val_loss가 줄어들지 않으면 learning rate를 auto-tune.
   - early stopping 대신 checkpointing 도입, best val_loss 지점에서 모델 저장.

3. **클래스 불균형 개선**  
   - class_weight 자동 계산 적용.
   - unknown_or_empty에 가중치 증가 → 해당 클래스 recall 증가 기대.

4. **Dropout, Regularization**  
   - Dropout, L2 Regularization으로 과적합 억제.

5. **데이터 분리의 정밀성**  
   - train/validation set이 카메라 환경 변화에도 일반화할 수 있도록, stratified split + randomness 보장.

6. **실험 결과의 체계적 기록**  
   - CSVLogger로 모든 epoch의 지표 기록.
   - 최종 best model 결과는 각각 history_colab.csv, random_test_results.csv로 저장.

---

## [CODE]

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Fixed base directory
base_dir = '/content/chess_dataset_colab'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')

img_size = (224, 224)
batch_size = 32
num_epochs = 50

# Prepare train and validation data generators with strong augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.10,
    zoom_range=[0.8, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=30.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Build model using pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True  # Fine-tune entire network

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Class weight computation
class_indices = train_generator.class_indices
class_labels_ordered = list(class_indices.keys())
y_train = train_generator.classes
weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), weights))

# Compile
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_cb = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False
)
csv_logger_cb = CSVLogger('history_colab.csv', append=False)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=4,
    min_lr=1e-6,
    verbose=1
)
callbacks = [checkpoint_cb, csv_logger_cb, reduce_lr_cb]

# Training
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Save training log as history_colab.csv (redundant because CSVLogger does this, but for clarity)
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_colab.csv', index=False)

# Evaluation on random test set
random_test_gen = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=img_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Load best weights
model.load_weights('best_model.h5')

# Predict
random_test_gen.reset()
preds = model.predict(random_test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = random_test_gen.classes

class_labels = list(random_test_gen.class_indices.keys())

# Classification report & confusion matrix
report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('random_test_results.csv')

# Optionally, print metrics for unknown_or_empty if present
if "unknown_or_empty" in class_labels:
    idx = class_labels.index("unknown_or_empty")
    print("unknown_or_empty recall:", report_df.loc["unknown_or_empty", "recall"])

```

---

### 주요 코드 적용 사항  
- **강화된 augmentation**: 실제 카메라 노이즈 및 왜곡에 근접하도록 설정.
- **클래스 불균형 자동 보정**: `compute_class_weight`로 unknown/empty 클래스 성능 개선 기대.
- **MobileNetV2 사전학습, fine-tuning**: base_model.trainable = True.
- **Dropout/L2 정규화**: 과적합 방지.
- **ReduceLROnPlateau**: overfitting·loss plateu 시 learning rate 감소.
- **CSVLogger, ModelCheckpoint, best model 평가 및 기록**: reproducibility & 실험 효율.

참고로, 경로 지정, 학습 로그 등은 반드시 실행 환경에 맞게 추가적으로 확인해 주세요.