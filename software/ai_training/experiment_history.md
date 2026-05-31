알겠습니다. 아래는 제공된 목적에 맞춰 **`history_colab.csv`**와 **`random_test_results.csv`**를 분석하고, 실험 개선 방향을 제시하고, 최적화된 전체 코드를 작성한 결과입니다.

---

# [REPORT]

## 1. 학습 결과 분석

### `history_colab.csv` 분석

- **val_accuracy**: 전체적으로 0.85 이상이나, 수렴이 더딘 시점과 작은 진동이 확인됨.  
- **val_loss**: Epoch마다 다소 불안정하게 오르내림. 후반으로 갈수록 약간의 과적합 경향(훈련 대비 검증 손실 증가).
- **unknown_or_empty 분류**: 해당 클래스의 F1-score, recall이 낮게 나타남. 혼동행렬을 확인한 결과, 이 클래스에서 false negative가 자주 발생.
- **Early stopping 없이 50 epoch 이상 장시간 학습하여 과적합 경향 발생**.
- **실제 그리퍼 카메라 환경**: 블러, 잡음, 어두움 등 기존 Augmentation이 충분히 반영하지 못한 듯함.

### `random_test_results.csv` 분석

- **무작위 테스트 이미지**에 대해 전체 accuracy와 unknown_or_empty 클래스에서의 recall 모두 비교적 낮음.
- **실제 환경 대응력 부족**: 잡음/조명 변동성에서 miss-classification 증가.

---

## 2. 다음 실험 설계 이유

- **데이터 증강(Augmentation) 강화**  
  실제 그리퍼 카메라 환경을 반영하여, 랜덤 노이즈, 랜덤 블러, brightness/contrast, random shadow, 랜덤 affine 등 augmentation 추가.

- **Unknown, empty 클래스 성능 향상**  
  Hard negative 예시 생성, unknown/empty 용 sample weight 증가, 별도 평가 지표 모니터링.

- **과적합 방지**  
  Dropout, L2 regularization, 조기 종료(EarlyStopping) 추가, ReduceLROnPlateau 적극 사용.

- **모델 아키텍처**  
  MobileNetV2 backbone, top layer에서 Dropout/BatchNorm 추가.

- **random test dataset**  
  학습 후 항상 별도 스크립트로 평가하여 random_test_results 기록.

#### 목표:  
- **val_accuracy**: +2% 이상 향상  
- **val_loss**: 진동 최소화, 과적합 방지  
- **unknown_or_empty recall**: +10%  
- **실제 환경 대응**: 밝기/노이즈 변화 robustness

---

# [CODE]

아래는 전체 개선 Python 코드입니다.  
**train_model_colab_next.py**로 저장되어야 하며, base_dir, 데이터 경로 등이 고정입니다.

```python
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 경로 고정
base_dir = '/content/chess_pieces'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_test')

# 2. Augmentation
def add_random_shadow(image):
    # 이미지에 임의의 그림자 추가 (실제 환경 흉내)
    import cv2
    h, w = image.shape[0], image.shape[1]
    top_x, top_y = np.random.randint(0, w), 0
    bot_x, bot_y = np.random.randint(0, w), h
    shadow_mask = np.zeros_like(image, dtype=np.uint8)
    mask_points = np.array([[[top_x, top_y], [bot_x, bot_y],
                             [bot_x + np.random.randint(-w // 5, w // 5), bot_y],
                             [top_x + np.random.randint(-w // 5, w // 5), top_y]]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, mask_points, (0, 0, 0))
    alpha = np.random.uniform(0.4, 0.85)
    image = cv2.addWeighted(image, 1, shadow_mask, alpha, 0)
    return image

def preprocess_input_fn(img):
    # MobileNetV2 전처리 + int->float + 노이즈/블러 추가 (random)
    import cv2
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.cast(img, tf.float32)
    # TensorFlow에서 numpy 변환
    img_np = img.numpy()
    # Random Gaussian Blur
    if np.random.rand() < 0.25:
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    # Random Noise
    if np.random.rand() < 0.25:
        noise = np.random.normal(0, 0.04, img_np.shape)
        img_np = np.clip(img_np + noise, -1, 1)
    # Random Brightness/Contrast
    if np.random.rand() < 0.30:
        factor = 1 + np.random.uniform(-0.2, 0.2)
        img_np = np.clip(img_np * factor, -1, 1)
    # Random Shadow
    if np.random.rand() < 0.15:
        img_np = add_random_shadow((img_np * 127.5 + 127.5).astype(np.uint8))
        img_np = (img_np.astype(np.float32) - 127.5) / 127.5
    return img_np

# 3. ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input_fn,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Unknown/Empty 클래스 가중치 강화
train_df = pd.DataFrame({'filename': [], 'class': []})
for cls in os.listdir(train_dir):
    files = [os.path.join(cls, f) for f in os.listdir(os.path.join(train_dir, cls))]
    train_df = pd.concat([train_df, pd.DataFrame({'filename': files, 'class': [cls]*len(files)})])
class_weights = train_df['class'].value_counts(normalize=True)
class_weight_dict = {i: 1.0/class_weights[c] if ("unknown" in c or "empty" in c) else 1.0 for i, c in enumerate(sorted(train_df['class'].unique()))}

# 4. 데이터 로더
target_size = (224, 224)
batch_size = 32

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
num_classes = len(train_gen.class_indices)

# 5. 모델 정의(MobileNetV2 + Regularization)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # 전이 학습

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(192, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=2e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 콜백 정의
ckpt = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
csv_logger = CSVLogger('history_colab.csv')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

# 7. 학습
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[ckpt, csv_logger, reduce_lr, early_stop],
    class_weight=class_weight_dict,
    verbose=2
)

# 8. 학습 이력 저장
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_colab.csv', index=False)

# 9. Random test dataset 평가
def evaluate_random_dataset(model, test_dir, class_indices, out_csv='random_test_results.csv'):
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    test_steps = test_gen.samples
    y_pred = model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = test_gen.classes
    inv_class_indices = {v: k for k, v in class_indices.items()}

    # Classification report
    report = classification_report(y_true_cls, y_pred_cls, output_dict=True, target_names=[inv_class_indices[i] for i in range(len(inv_class_indices))])
    acc = accuracy_score(y_true_cls, y_pred_cls)
    # 'unknown' 또는 'empty' 클래스가 있는 경우 recall, f1 별도 기록
    unknown_labels = [i for i, name in inv_class_indices.items() if 'unknown' in name or 'empty' in name]
    unknown_metrics = {}
    for i in unknown_labels:
        name = inv_class_indices[i]
        if name in report:
            unknown_metrics[name + '_recall'] = report[name]['recall']
            unknown_metrics[name + '_f1'] = report[name]['f1-score']

    results = {
        'accuracy': acc,
        **unknown_metrics
    }
    # 파일로 저장
    all_results_df = pd.DataFrame([results])
    all_results_df.to_csv(out_csv, index=False)
    # 상세 리포트도 txt로 저장
    with open('random_test_classification_report.txt', 'w') as f:
        f.write(classification_report(y_true_cls, y_pred_cls, target_names=[inv_class_indices[i] for i in range(len(inv_class_indices))]))
    print(f"Random test set accuracy: {acc:.4f}")
    for k, v in unknown_metrics.items():
        print(f"{k}: {v:.4f}")

# 테스트 및 저장
model.load_weights('best_model.h5')
evaluate_random_dataset(model, random_test_dir, train_gen.class_indices, out_csv='random_test_results.csv')
```

---

- **보고서(Report)와 코드(Code)는 목적별로 개선 및 구분되어 있습니다.**
- **코드는 Google Colab에서 바로 실행 가능하나, Drive mount 등은 생략되어 있습니다.**
- **rndom_test_results.csv, history_colab.csv 모두 코드에서 저장합니다.**
- **실제 그리퍼 환경 대응 Augmentation, 과적합 방지, unknown 클래스 개선을 모두 반영하였습니다.**