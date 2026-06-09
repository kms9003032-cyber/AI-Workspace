import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42
MODEL_PATH = 'best_model_colab.keras'
FINAL_MODEL_PATH = 'final_model_colab.keras'
CSV_LOGGER_PATH = 'history_colab.csv'
RANDOM_TEST_CSV = 'random_test_results.csv'
experiment_history = []

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=0.18,
    brightness_range=[0.65, 1.35],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED,
    shuffle=True
)
val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

labels = train_gen.classes
n_classes = train_gen.num_classes
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
try:
    class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight = dict(enumerate(class_weights_arr))
except Exception as e:
    class_weight = None

def build_model(n_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    output = Dense(n_classes, activation='softmax', kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

model = None
load_failed_reason = ""
if os.path.isfile(MODEL_PATH):
    try:
        loaded_model = load_model(MODEL_PATH)
        output_shape = loaded_model.output_shape[-1]
        if output_shape == n_classes:
            model = loaded_model
            experiment_history.append({'resume': True, 'reason': 'best_model_colab.keras 불러오기 성공'})
        else:
            load_failed_reason = f"기존 모델 클래스수({output_shape}) != 현 클래스수({n_classes})"
            model = build_model(n_classes)
            experiment_history.append({'resume': False, 'reason': load_failed_reason})
    except Exception as ex:
        load_failed_reason = str(ex)
        model = build_model(n_classes)
        experiment_history.append({'resume': False, 'reason': load_failed_reason})
else:
    model = build_model(n_classes)
    experiment_history.append({'resume': False, 'reason': 'best_model_colab.keras 없음'})

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    MODEL_PATH, monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max'
)
csv_logger_cb = CSVLogger(CSV_LOGGER_PATH, append=True)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss', factor=0.4, patience=7, verbose=1, min_lr=2e-6
)
callbacks = [checkpoint_cb, csv_logger_cb, reduce_lr_cb]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)
model.save(FINAL_MODEL_PATH)

if os.path.isdir(raw_test_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = []
    image_filenames = []
    for fname in os.listdir(raw_test_dir):
        if any(fname.lower().endswith(e) for e in valid_exts):
            image_filenames.append(fname)
    results = []
    loaded_best_model = None
    try:
        loaded_best_model = load_model(MODEL_PATH)
    except:
        loaded_best_model = model
    for fname in image_filenames:
        path = os.path.join(raw_test_dir, fname)
        try:
            img = load_img(path, target_size=IMG_SIZE)
            x = img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            preds = loaded_best_model.predict(x)
            top_idx = np.argmax(preds[0])
            confidence = float(preds[0][top_idx])
            pred_label = idx_to_class[top_idx]
            results.append({'filename': fname, 'pred_label': pred_label, 'confidence': confidence})
        except Exception as e:
            results.append({'filename': fname, 'pred_label': 'ERROR', 'confidence': 0.0})
    pd.DataFrame(results).to_csv(RANDOM_TEST_CSV, index=False)
else:
    experiment_history.append({'raw_test_evaluation': False, 'reason': f"{raw_test_dir} 폴더 없음"})

try:
    if len(experiment_history) > 0:
        pd.DataFrame(experiment_history).to_csv('experiment_history.csv', index=False)
except:
    pass