import os
import random
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import classification_report

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
best_model_path = '/content/best_model_colab.keras'
final_model_path = '/content/final_model_colab.keras'
csv_logger_path = '/content/history_colab.csv'
random_test_results_path = '/content/random_test_results.csv'
experiment_history_path = '/content/experiment_history.log'

img_size = 224
batch_size = 32
seed = 777

def get_classes(d):
    return sorted([folder for folder in os.listdir(d) if os.path.isdir(os.path.join(d, folder))])

def get_class_counts(d, classes):
    counts = []
    for c in classes:
        counts.append(len(glob(os.path.join(d, c, '*.*'))))
    return counts

classes = get_classes(train_dir)
n_classes = len(classes)
class_counts = get_class_counts(train_dir, classes)

min_count = np.min(class_counts)
class_weight = {i: min_count/class_counts[i] if class_counts[i]>0 else 1.0 for i in range(n_classes)}
unknown_idx = [i for i, c in enumerate(classes) if 'unknown' in c.lower() or 'empty' in c.lower()]
for idx in unknown_idx:
    class_weight[idx] *= 1.6

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.12,
    height_shift_range=0.12,
    brightness_range=(0.5, 1.6),
    shear_range=12,
    zoom_range=0.15,
    channel_shift_range=28,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

def build_model(n_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = True
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def log_experiment(msg):
    with open(experiment_history_path, 'a') as f:
        f.write(msg+'\n')

model = None
previous_classes = None
need_new_model = False
if os.path.exists(best_model_path):
    try:
        model = load_model(best_model_path)
        output_shape = model.output_shape[-1]
        if output_shape != n_classes:
            need_new_model = True
            log_experiment(f"모델 이어학습 불가: 클래스수 불일치 (기존:{output_shape}, 현재:{n_classes})")
        else:
            previous_classes = None
    except Exception as e:
        need_new_model = True
        log_experiment(f"모델 이어학습 불가: 로드 에러 {e}")
else:
    need_new_model = True
    log_experiment("최초 학습: best_model_colab.keras 없음. MobileNetV2 새로 생성.")

if need_new_model:
    model = build_model(n_classes)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=6,
    min_lr=3e-6,
    verbose=1
)
csv_logger = CSVLogger(csv_logger_path)

epochs = 100
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, csv_logger, reduce_lr],
    class_weight=class_weight,
    verbose=1
)
model.save(final_model_path)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(csv_logger_path, index=False)

if os.path.exists(best_model_path):
    model = load_model(best_model_path)

if os.path.exists(raw_test_dir) and os.path.isdir(raw_test_dir):
    img_files = [f for f in os.listdir(raw_test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(img_files) > 0:
        idx2class = {i:c for i,c in enumerate(classes)}
        results = []
        for f in img_files:
            fp = os.path.join(raw_test_dir, f)
            try:
                img = load_img(fp, target_size=(img_size, img_size))
                arr = img_to_array(img)
                arr = arr/255.0
                arr = np.expand_dims(arr, axis=0)
                preds = model.predict(arr)
                pred_idx = np.argmax(preds[0])
                conf = float(np.max(preds[0]))
                pred_class = idx2class[pred_idx]
                results.append({
                    'filename':f,
                    'pred_class':pred_class,
                    'confidence':conf
                })
            except Exception as e:
                results.append({
                    'filename':f,
                    'pred_class':'error',
                    'confidence':-1
                })
        results_df = pd.DataFrame(results)
        results_df.to_csv(random_test_results_path, index=False)
else:
    log_experiment("raw_dataset 존재하지 않아 평가 skip.")