import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import albumentations as A
from albumentations.experimental import Compose
from albumentations.core.composition import OneOf
import cv2
from tensorflow.keras.utils import Sequence

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100

history_csv_path = 'history_colab.csv'
best_model_path = 'best_model_colab.keras'
final_model_path = 'final_model_colab.keras'
random_test_results_csv = 'random_test_results.csv'

experiment_history = []

class CustomAumentedSequence(Sequence):
    def __init__(self, image_gen, augment, batch_size):
        self.image_gen = image_gen
        self.augment = augment
        self.batch_size = batch_size
        self.on_epoch_end()
    def __len__(self):
        return len(self.image_gen)
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_gen.filenames))
        if self.image_gen.shuffle:
            np.random.shuffle(self.indexes)
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for i in indexes:
            filename = os.path.join(self.image_gen.directory, self.image_gen.filenames[i])
            image = cv2.imread(filename)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_SIZE)
                if self.augment is not None:
                    image = self.augment(image=image)['image']
                image = image.astype(np.float32) / 255.0
                batch_images.append(image)
                batch_labels.append(self.image_gen.labels[i])
        return np.array(batch_images), keras.utils.to_categorical(batch_labels, num_classes=len(self.image_gen.class_indices))

albu_transform = A.Compose([
    A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT, p=0.95),
    A.HorizontalFlip(p=0.95),
    A.VerticalFlip(p=0.95),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.9),
    A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=18, val_shift_limit=18, p=0.7),
    A.Blur(blur_limit=5, p=0.3),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
    A.RandomResizedCrop(IMG_SIZE[0], IMG_SIZE[1], scale=(0.7,1.0), p=0.2),
    A.Cutout(max_h_size=IMG_SIZE[0]//6, max_w_size=IMG_SIZE[1]//6, fill_value=0, p=0.35),
    A.CoarseDropout(max_holes=3, max_height=IMG_SIZE[0]//9, max_width=IMG_SIZE[1]//9, p=0.25)
])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_img_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_img_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_img_gen.class_indices)
y_train = train_img_gen.classes
class_weights_arr = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
class_weight_dict = {i: cw for i, cw in enumerate(class_weights_arr)}

train_gen = CustomAumentedSequence(train_img_gen, augment=albu_transform, batch_size=BATCH_SIZE)
val_gen = CustomAumentedSequence(val_img_gen, augment=None, batch_size=BATCH_SIZE)

original_model_loaded = False
model = None

if os.path.exists(best_model_path):
    try:
        loaded_model = keras.models.load_model(best_model_path)
        last_layer_units = loaded_model.output_shape[-1]
        if last_layer_units == num_classes:
            model = loaded_model
            original_model_loaded = True
            experiment_history.append('load_model: 기존 best_model_colab.keras로 이어학습 (클래스수 일치)')
            print('기존 모델에서 이어학습')
        else:
            experiment_history.append('새 모델: 클래스수 변경됨 (이전: %d, 현재:%d). 신규 MobileNetV2 생성.' % (last_layer_units, num_classes))
    except Exception as e:
        experiment_history.append('새 모델: best_model_colab.keras 구조 불일치. 신규 MobileNetV2 생성.')
if model is None:
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = True
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=True)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    experiment_history.append('새 MobileNetV2 모델 생성')
optimizer = keras.optimizers.Adam(learning_rate=5e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
lr_reduce_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=7,
    min_lr=1e-6,
    mode='min',
    verbose=1
)
csv_logger_cb = CSVLogger(history_csv_path)
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    restore_best_weights=True
)

callbacks = [checkpoint_cb, lr_reduce_cb, csv_logger_cb, earlystop_cb]

steps_per_epoch = len(train_gen)
validation_steps = len(val_gen)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

model.save(final_model_path)

if os.path.exists(history_csv_path) and os.path.isfile(history_csv_path):
    try:
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(history_csv_path, mode='w', header=True, index=False)
    except:
        pass

if os.path.exists(raw_test_dir) and os.path.isdir(raw_test_dir):
    ext_list = ['.jpg', '.jpeg', '.png']
    raw_img_files = [f for f in os.listdir(raw_test_dir) if os.path.splitext(f)[1].lower() in ext_list]
    results = []
    for img_fn in raw_img_files:
        img_path = os.path.join(raw_test_dir, img_fn)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            arr = img.astype(np.float32) / 255.0
            pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
            pred_class_idx = np.argmax(pred)
            pred_class_label = [k for k, v in train_img_gen.class_indices.items() if v == pred_class_idx][0]
            confidence = float(pred[pred_class_idx])
            label = pred_class_label
        except Exception as e:
            label = 'predict_error'
            confidence = -1
        results.append({'filename': img_fn, 'pred_class': label, 'confidence': confidence})
    if len(results) > 0:
        pd.DataFrame(results).to_csv(random_test_results_csv, index=False)
else:
    experiment_history.append('raw_dataset 평가 스킵: 폴더 없음 또는 비어 있음')

if len(experiment_history) > 0:
    exp_log_path = 'experiment_history.txt'
    with open(exp_log_path, 'a') as f:
        for line in experiment_history:
            f.write(str(line) + '\n')