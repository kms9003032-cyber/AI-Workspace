import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report
import traceback

BASE_DIR = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
RAW_TEST_DIR = os.path.join(BASE_DIR, 'raw_dataset')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model_colab.keras')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'final_model_colab.keras')
CSV_LOGGER_PATH = os.path.join(MODELS_DIR, 'history_colab.csv')
TEST_RESULTS_PATH = os.path.join(MODELS_DIR, 'random_test_results.csv')
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)

experiment_report = []
initial_raw_acc = None
previous_best_acc = None

try:
    if os.path.exists(BEST_MODEL_PATH):
        tmp_model = load_model(BEST_MODEL_PATH)
        base_model_classes = tmp_model.layers[-1].output_shape[-1]
        tmp_datagen = ImageDataGenerator(rescale=1./255)
        tmp_gen = tmp_datagen.flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
        if base_model_classes != tmp_gen.num_classes:
            raise ValueError('불일치 클래스 수')
        model = load_model(BEST_MODEL_PATH)
        experiment_report.append('INFO: 기존 best_model_colab.keras에서 이어학습 진행')
    else:
        raise FileNotFoundError
except Exception as e:
    experiment_report.append(f'INFO: best_model_colab.keras 이어학습 실패: {str(e)} / {traceback.format_exc()}')
    base_model = MobileNetV2(
        input_shape=IMG_SIZE+(3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(len(os.listdir(TRAIN_DIR)), activation='softmax')
    ])
    experiment_report.append('INFO: 새 MobileNetV2 구조로 학습 시작')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=(0.60,1.40),
    shear_range=10.0,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=None
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

NUM_CLASSES = train_generator.num_classes

if not isinstance(model, tf.keras.Sequential) or model.layers[-1].output_shape[-1] != NUM_CLASSES:
    base_model = MobileNetV2(
        input_shape=IMG_SIZE+(3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    experiment_report.append('INFO: 출력 클래스 재조정 - 새 모델로 재시작')

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0007),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csv_logger_cb = callbacks.CSVLogger(CSV_LOGGER_PATH, append=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=6,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, csv_logger_cb, reduce_lr_cb]
)

model.save(FINAL_MODEL_PATH)

try:
    if os.path.isdir(RAW_TEST_DIR):
        image_files = [
            fname for fname in os.listdir(RAW_TEST_DIR)
            if any(fname.lower().endswith(ext) for ext in ['jpg','jpeg','png'])
        ]
        if len(image_files)>0:
            best_model = load_model(BEST_MODEL_PATH)
            label_map = train_generator.class_indices
            label_map_rev = {v:k for k,v in label_map.items()}
            results = []
            acc_count = 0
            total = len(image_files)
            for fname in image_files:
                img_path = os.path.join(RAW_TEST_DIR, fname)
                try:
                    img = load_img(img_path, target_size=IMG_SIZE)
                    arr = img_to_array(img) / 255.
                    pred = best_model.predict(np.expand_dims(arr, axis=0), verbose=0)
                    pred_idx = np.argmax(pred)
                    confidence = float(np.max(pred))
                    pred_label = label_map_rev[pred_idx]
                except Exception as e:
                    pred_label='error'
                    confidence=0.
                results.append({'filename':fname, 'pred_label':pred_label, 'confidence':confidence})
            df_results = pd.DataFrame(results)
            df_results.to_csv(TEST_RESULTS_PATH, index=False)
            majority_label = df_results['pred_label'].mode()[0] if len(df_results)>0 else ''
            raw_acc = (df_results['pred_label'] == majority_label).sum()/len(df_results) if len(df_results)>0 else 0
            experiment_report.append(f'RAW DATASET 평가: majority_label "{majority_label}", 정확도: {raw_acc:.4f}')
        else:
            experiment_report.append('INFO: raw_dataset 이미지가 없음 (RAW 평가 스킵)')
    else:
        experiment_report.append('INFO: raw_dataset 폴더 없음 (RAW 평가 스킵)')
except Exception as e:
    experiment_report.append('ERROR: raw_dataset 평가 오류 - '+str(e))

try:
    with open(os.path.join(MODELS_DIR, 'experiment_report.txt'), 'w') as f:
        for line in experiment_report:
            f.write(line+'\n')
except Exception:
    pass