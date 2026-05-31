[REPORT]

## 1. 학습 결과 분석

### history_colab.csv 분석

학습 결과를 기반으로 다음과 같은 주요 관찰 결과가 도출되었습니다:

- **val_accuracy**가 일정 epoch 이후 상승이 정체 또는 하락(plateau/overfit)하는 경향이 보입니다.
- **val_loss**가 불규칙적으로 증가하거나, train_loss와의 간격이 넓어지면서 overfitting 신호가 명확합니다.
- **unknown_or_empty**(추정상, 분류 불가 class 또는 예측 없음)의 비율이 유지되거나, 증가하는 구간이 있습니다.
- 데이터 다양성이 부족하거나 augmentation이 부족한 신호, 또는 실제 환경(카메라/그리퍼) 데이터와 분포 차이가 클 수 있습니다.

### random_test_results.csv 분석

- 실제 "random dataset"에서의 성능(val과의 차이)에서 accuracy 하락 및 unknown/empty 비율이 증가합니다.
- 실제 환경(조명, noise, object pose)의 차이로 인해, 학습된 모델의 일반화 성능이 한계에 부딪힘을 알 수 있습니다.
- 과적합 및 domain gap 해소가 필요한 증거입니다.

## 2. 다음 실험 설계 이유

- **augmentation 강화**: 밝기, 대비, noise, 좌우반전, shift/scale 등 실제 환경 다양성 반영! 기존 augmentation을 늘려 일반화 강화.
- **MobileNetV2**: 파라미터 수는 적으나 충분한 표현력을 가질 수 있고, 카메라 환경에 적합한 속도/경량 모델이므로 유지. 단, tf.keras.applications의 pre-trained weights 이용(fine-tune은 frozen layer 개수 조절).
- **ModelCheckpoint, ReduceLROnPlateau, EarlyStopping** 등으로 unstable val_loss 대응 및 과적합 방지.
- **unknown_or_empty** 개선: 데이터 imbalance 해결(클래스 별 weight, oversample), prediction threshold 재설정 등 도입.
- **실제 환경(카메라/그리퍼)** 대응: train/val 데이터에 노이즈, artifact, 조도, blurring 등 환경효과 추가.
- **모델 저장 및 logger**: 결과 재현성 유지 및 이력 확인.

위 방법을 반영한 새 코드를 아래에 제안합니다.

---

[CODE]

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import pandas as pd

# === base_dir 및 데이터 경로 고정 ===
base_dir = '/content/drive/MyDrive/chess_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')

# === Hyperparameters ===
img_size = 224
batch_size = 32
epochs = 60
lr = 1e-4
num_classes = len(os.listdir(train_dir))  # assumes one folder per class

# === 데이터 augmentation 설정 ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.15,
    zoom_range=[0.85, 1.15],
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# === 데이터 generator ===
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === 모델 정의 (MobileNetV2) ===
base_model = MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)

# 상위 일부 layer만 fine-tune
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

opt = Adam(learning_rate=lr)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# === 콜백 ===
checkpoint_cb = ModelCheckpoint(
    os.path.join(base_dir, 'best_model.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
csvlogger_cb = CSVLogger(os.path.join(base_dir, 'history_colab.csv'), append=False)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# === 클래스별 sample 수 불균형 correction용 class_weight ===
class_counts = train_generator.classes
_, counts = np.unique(class_counts, return_counts=True)
class_weight = dict(enumerate(np.max(counts) / counts))

# === 학습 ===
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint_cb, csvlogger_cb, reduce_lr_cb, earlystop_cb],
    class_weight=class_weight
)

# === 랜덤테스트 데이터셋 평가 ===
random_test_generator = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_size, img_size),
    batch_size=1,  # 하나씩 평가
    class_mode='categorical',
    shuffle=False
)

random_test_steps = random_test_generator.samples

# 예측 및 통계
preds = model.predict(random_test_generator, steps=random_test_steps, verbose=1)
y_true = random_test_generator.classes
y_pred = np.argmax(preds, axis=1)
class_indices_rev = {v: k for k, v in random_test_generator.class_indices.items()}

# unknown_or_empty 예측: 확률이 max값이 특정 임계치(예: 0.5)보다 작은 경우 unknown 처리 (예시)
threshold = 0.5
unknown_pred = np.max(preds, axis=1) < threshold
y_pred_final = [y if not unk else -1 for y, unk in zip(y_pred, unknown_pred)]

# 평가 통계 계산
accuracy = np.mean(np.array(y_pred_final) == y_true)
unknown_count = np.sum(unknown_pred)
result_rows = []

for idx, (true_label, pred, pred_prob, is_unk) in enumerate(zip(y_true, y_pred_final, np.max(preds, axis=1), unknown_pred)):
    result_rows.append({
        "filename": random_test_generator.filenames[idx],
        "true_class": class_indices_rev[true_label] if true_label in class_indices_rev else 'unknown',
        "pred_class": class_indices_rev[pred] if pred >= 0 and pred in class_indices_rev else 'unknown_or_empty',
        "pred_prob": pred_prob,
        "is_unknown_or_empty": int(is_unk)
    })

df = pd.DataFrame(result_rows)
df.to_csv(os.path.join(base_dir, 'random_test_results.csv'), index=False)

print(f"Random test accuracy: {accuracy:.4f}")
print(f"Unknown/empty ratio: {unknown_count/len(y_true):.4f}")

# === 끝 ===
```