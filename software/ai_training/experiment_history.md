물론입니다! 아래는 주어진 목표와 파일에 기반하여 **학습 결과를 분석**하고, **다음 실험 방안**을 제안한 리포트와, 해당 전략을 반영한 개선된 코드입니다.

---

# [REPORT]

## 1. 학습 결과 분석

- **val_accuracy**:  
  결과 파일에 의하면 `val_accuracy`는 꾸준히 증가하다가 중반 이후 정체/하락 현상이 나타났으며, 훈련 정확도와 다소 차이가 있다. 이는 일반화에 한계가 있음을 시사한다.

- **val_loss**:  
  `val_loss`는 초반 급격히 감소하다가 이후 등락이 반복되며 상대적으로 불안정한 모습을 보인다. 이는 오버피팅 또는 validation set과 train set의 분포 차이, batch normalization 미 최적화 등이 원인이 될 수 있다.

- **unknown_or_empty 개선 필요**:  
  별도의 `unknown_or_empty` category(클래스)가 오분류되는 경우가 많고, 이 카테고리의 recall이 낮다. 이는 데이터 부족, 클래스 불균형, No-object 샘플의 환경 다양성 부족, augmentation 미흡 등에서 기인할 수 있다.

- **실제 환경(그리퍼 카메라) 대응**:  
  random_test에서 실제 환경 이미지를 평가한 결과, 실제 환경에서의 정확도가 validation set보다 낮게 나타났다. 카메라 노이즈, 조명, 배경 등 환경 변화에 모델이 robust하지 않음을 보여준다.

- **과적합**:  
  학습/검증 정확도 격차와 validation loss 불안정성은 오버피팅 가능성을 보여준다. Dropout, Regularization, Augmentation 강화, epochs 조정, Early Stopping 적용 등이 해결책이다.


## 2. 다음 실험 설계 및 이유

- **모델**:  
  효율적 파라미터(모바일 환경 고려)의 MobileNetV2를 유지하되, input shape 및 layer unfreeze 범위만 조정한다.

- **Augmentation**:  
  실제 환경 적응력(조명, 블러, 노이즈, color jitter 등) 강화. 특히 No-object에선 강한 augmentation 적용.

- **Class weight & Oversampling**:  
  unknown_or_empty 클래스의 레코드 비중이 낮아서 class_weights 적용 및 oversampling 시도.

- **Validation split 교차성 확보**:  
  Stratified split, validation set에 다양한 카메라 환경 반영.

- **Regularization**:  
  Dropout 추가, ReduceLROnPlateau 모니터링 강화로 일반화 확보.

- **학습 스케줄**:  
  ModelCheckpoint(val_loss 기준), CSVLogger, 얼리 스톱핑, epoch 조정.

- **테스트 평가**:  
  랜덤 샘플(랜덤 각도, 조명 등)에서 성능 측정, 세부 confusion matrix 기록.

# [CODE]

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# base_dir 설정
base_dir = '/content/data/chess_dataset'  # dataset 베이스 경로

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50
RANDOM_SEED = 42

# 클래스 이름 추출
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
n_classes = len(class_names)

# 클래스 불균형 보정: 클래스별 샘플 수 카운트
def get_class_counts(directory):
    labels = []
    for cl in class_names:
        cl_dir = os.path.join(directory, cl)
        labels += [cl] * len(os.listdir(cl_dir))
    return pd.Series(labels).value_counts().to_dict()

class_counts = get_class_counts(train_dir)
labels_list = []
for cl in class_names:
    labels_list += [cl] * class_counts[cl]
class_weights = compute_class_weight('balanced', classes=np.array(class_names), y=labels_list)
class_weights = dict(zip(range(n_classes), class_weights))

# 강한 Augmentation
train_aug = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=(0.7, 1.3),
    channel_shift_range=25.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=lambda x: tf.image.random_jpeg_quality(x, 85, 100),
)

val_aug = ImageDataGenerator(
    rescale=1/255.
)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_SEED
)

val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# MobileNetV2 with partial fine-tuning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    if layer.name.startswith('block_13'):
        set_trainable = True
    layer.trainable = set_trainable  # 마지막 블록만 fine-tune

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(n_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 콜백 선언
callbacks = [
    ModelCheckpoint('best_model_chess.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    CSVLogger('history_colab.csv'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
]

# 학습
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# csv로그는 이미 자동 저장됨(CSVLogger).

# 랜덤 테스트셋 평가용
random_test_dir = os.path.join(base_dir, 'random_dataset')
random_gen = val_aug.flow_from_directory(
    random_test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
random_pred = model.predict(random_gen, verbose=1)
random_pred_labels = np.argmax(random_pred, axis=1)
random_true_labels = random_gen.classes

results_df = pd.DataFrame({
    'filename': [os.path.basename(fname) for fname in random_gen.filenames],
    'true': [class_names[i] for i in random_true_labels],
    'pred': [class_names[i] for i in random_pred_labels]
})
results_df['correct'] = results_df['true'] == results_df['pred']
results_df.to_csv('random_test_results.csv', index=False)
```

---

이 코드는 체스말 분류 AI 실험에 바로 사용할 수 있게 설계되었습니다.  
다음 단계에서 모델의 일반화와 실제 환경 적응 능력이 한층 강화될 것입니다.