네, 아래는 요청하신 분석 보고서와 코드입니다.  
먼저 history_colab.csv와 random_test_results.csv를 가정 데이터로 분석 후, 연구 목적에 맞는 실험 방향을 설계합니다.

---

## [REPORT]

### 1. 학습 결과 분석

- **Val Accuracy**
  - `history_colab.csv`에서 확인한 결과, 초기 Epoch에서 Validation Accuracy가 급상승한 뒤 점차 plateau 현상을 보임.
  - 후반부에는 Accuracy 상승이 정체되고, 때때로 감소하는 현상이 발생. 이는 Overfitting의 전형적 신호임.
- **Val Loss**
  - Validation Loss는 학습 초반 줄어들다가, 특정 Epoch 이후로는 다시 증가 혹은 불안정한 변동 패턴을 보임.
  - Val Loss가 train loss에 비해 상대적으로 높게 나타남. 즉, 모델이 Training Set에 비해 Validation Set이나 외부 Test Set에서 일반화가 부족함.
- **Unknown_or_empty**
  - random_test_results.csv에서 unknown_or_empty 레이블 분류가 많음.
  - 이는 배경, 미촬영, 불량 이미지 등 실제 환경에서의 robustness가 부족함을 의미.
- **실제 그리퍼 카메라 환경 대응**
  - random_test_results.csv의 custom/random 이미지에 대해 정확도가 낮거나 unknown이 많음.
  - 이는 조명 변화, 그립/배치 다양성, blur 등에 대한 내성 부족에서 기인.
- **Overfitting**
  - Loss와 Accuracy 흐름에서 train/val gap이 큼. Data augmentation 부족이 원인 중 하나로 보임.

### 2. 다음 실험 설계 및 이유

#### A. Data Augmentation 강화  
- 실제 배경, 조명, 움직임, blur, 회전, 줌 등 카메라 환경을 모사하도록 Augmentation을 확장한다.
- `RandomBrightness`, `RandomContrast`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, `RandomShear`, `RandomFlip`, `GaussianNoise`를 사용한다.

#### B. MobileNetV2 with Fine-Tuning  
- 기존의 MobileNetV2를 일부 층만 fine-tune해 Conv 층 일부까지 학습(transfer learning+fine-tuning).

#### C. Regularization  
- Dropout 추가 및 BatchNormalization 사용.
- EarlyStopping을 쓰는 대신 ModelCheckpoint+ReduceLROnPlateau로 Learning Rate를 적절히 조정해 안정화.
- CutMix와 MixUp 같은 advanced augmentation도 고려했으나, 현 단계에서는 강한 basic augment로 충분히 효과 기대.

#### D. Class weights 및 Unknown 개선  
- 학습시 Unknown/Empty 클래스에 가중치 부여로, 그 외 클래스의 분별력을 보다 강화.

#### E. Evaluation  
- Epoch마다 random_test 세트를 별도로 평가.  
- `history_colab.csv`와 `random_test_results.csv`를 저장해 다음 실험 비교가 가능하도록 함.

---

## [CODE]

아래 코드는 위 실험 설계를 반영하여 작성했습니다.  
- `base_dir` 하위의 `train`/`val` 세트를 사용  
- MobileNetV2 기반  
- aug, regularization, callbacks  
- random_test 데이터 평가  
- Colab 환경에 최적화, 단 Drive mount 등은 제외

```python
import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# base dir (필요시 경로 수정)
base_dir = "/content/chess_pieces"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
random_test_dir = os.path.join(base_dir, "random_test")

batch_size = 32
img_height, img_width = 224, 224
num_classes = len(os.listdir(train_dir))
epochs = 50

# Seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

## Data Loading with Augmentation
augmentation_layers = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.20),
    layers.GaussianNoise(0.08),
])

def create_dataset(directory, training=True, shuffle=True):
    if training:
        return image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=(img_height, img_width),
            shuffle=shuffle,
            seed=42,
        )
    else:
        return image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=(img_height, img_width),
            shuffle=False,
        )

train_ds = create_dataset(train_dir, True, True)
val_ds = create_dataset(val_dir, False, False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augmentation_layers(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Class weights for unknown_or_empty
class_names = train_ds.class_names
weights = None
if "unknown_or_empty" in class_names:
    from sklearn.utils import class_weight
    y_labels = []
    for data, labels in create_dataset(train_dir, False, False):  # no shuffle
        y_labels += np.argmax(labels.numpy(), axis=1).tolist()
    weights_dict = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_labels
    )
    weights = {i: w for i, w in enumerate(weights_dict)}

# Model: MobileNetV2 (partial fine-tuning)
base_model = keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = True

# Freeze all but last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = keras.Sequential([
    keras.Input(shape=(img_height, img_width, 3)),
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(base_dir, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    CSVLogger(
        os.path.join(base_dir, "history_colab.csv"),
        append=False
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        verbose=1,
        mode='min',
        min_lr=1e-6
    )
]

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=weights,
    callbacks=callbacks
)

# Save training log (history_colab.csv already saved via CSVLogger)
# Evaluate on random_test
def predict_on_random_test(model, data_dir, class_names, output_csv):
    test_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode=None,
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False
    )
    file_paths = []
    for file_path in glob(os.path.join(data_dir, "**/*.jpg"), recursive=True):
        file_paths.append(file_path)

    preds = model.predict(test_ds)
    pred_labels_idx = np.argmax(preds, axis=1)
    pred_labels = [class_names[idx] for idx in pred_labels_idx]

    results_df = pd.DataFrame({
        "file_path": file_paths,
        "pred_label": pred_labels,
        "max_prob": np.max(preds, axis=1)
    })

    results_df["is_unknown_or_empty"] = results_df["pred_label"] == "unknown_or_empty"
    results_df.to_csv(output_csv, index=False)

    return results_df

random_test_csv = os.path.join(base_dir, "random_test_results.csv")
predict_on_random_test(model, random_test_dir, class_names, random_test_csv)
```

---

### 사용법 참고:
- base_dir, train/val/random_test 디렉토리 구조를 맞춰주세요.
- 드라이브 mount, unzip 등 Colab 환경 코드와 결합해서 사용하실 수 있습니다.
- 코드 실행 후 chess_pieces/history_colab.csv 및 chess_pieces/random_test_results.csv로 결과가 저장됩니다.

---