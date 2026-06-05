import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import cv2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 777

def add_random_noise(img):
    noise = np.random.normal(0, 0.04, img.shape)
    img = img + noise
    img = np.clip(img, 0., 1.)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.17,
    brightness_range=(0.7, 1.3),
    channel_shift_range=40.,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=add_random_noise
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical',
    seed=SEED
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
class_names = list(class_indices.keys())

y_tr = train_gen.classes
class_weight_dict = dict(
    enumerate(
        compute_class_weight(class_weight='balanced', classes=np.unique(y_tr), y=y_tr)
    )
)

best_model_path = 'best_model_colab.keras'
final_model_path = 'final_model_colab.keras'
history_csv_path = 'history_colab.csv'
random_test_csv_path = 'random_test_results.csv'
experiment_history_path = 'experiment_history.txt'

def load_existing_model():
    if os.path.exists(best_model_path):
        try:
            loaded = load_model(best_model_path)
            if loaded.output_shape[-1] == num_classes:
                return loaded, True
        except Exception as e:
            with open(experiment_history_path, 'a') as f:
                f.write(f"모델 구조/클래스 불일치로 best_model_colab.keras 이어학습 실패.\n{str(e)}\n")
    return None, False

def build_new_model(num_classes):
    inputs = Input(shape=(224,224,3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model, loaded_flag = load_existing_model()
if not model:
    model = build_new_model(num_classes)
    with open(experiment_history_path, 'a') as f:
        f.write("기존 모델 불러오기 실패. 구조 또는 클래스 수 차이로 새 모델 시작.\n")
    loaded_flag = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mc = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csv_logger = CSVLogger(history_csv_path)
rlrop = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=8,
    verbose=1,
    min_lr=1e-6
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[mc, rlrop, csv_logger]
)

model.save(final_model_path)

if os.path.exists(best_model_path):
    model = load_model(best_model_path)

def is_image_file(name):
    ext = name.lower().split('.')[-1]
    return ext in ['jpg', 'jpeg', 'png']

if os.path.exists(raw_test_dir):
    filenames = [f for f in os.listdir(raw_test_dir) if is_image_file(f)]
    results = []
    for fname in filenames:
        path = os.path.join(raw_test_dir, fname)
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, IMAGE_SIZE)
            img_norm = img_resized.astype(np.float32) / 255.
            img_input = np.expand_dims(img_norm, axis=0)
            preds = model.predict(img_input)
            pred_idx = np.argmax(preds[0])
            pred_label = class_names[pred_idx]
            confidence = float(np.max(preds[0]))
            result_row = {
                'filename': fname,
                'predicted_class': pred_label,
                'confidence': confidence
            }
            prob_cols = {f'prob_{class_names[i]}': float(preds[0][i]) for i in range(num_classes)}
            result_row.update(prob_cols)
            results.append(result_row)
        except Exception as e:
            results.append({'filename': fname, 'predicted_class': 'error', 'confidence': 0.0})
    results_df = pd.DataFrame(results)
    results_df.to_csv(random_test_csv_path, index=False)
else:
    with open(experiment_history_path, 'a') as f:
        f.write("raw_dataset 폴더 없음: RAW 평가 건너뜀.\n")