import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pandas as pd
import random
from tensorflow.keras.utils import to_categorical

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
best_model_path = os.path.join(base_dir, 'best_model_colab.keras')
final_model_path = os.path.join(base_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(base_dir, 'history_colab.csv')
random_test_csv_path = os.path.join(base_dir, 'random_test_results.csv')
experiment_history = []

img_size = 224
batch_size = 32
epochs = 100
lr = 1e-4

def get_class_names_and_count(train_dir):
    class_names = [d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
    num_classes = len(class_names)
    return class_names, num_classes

class_names, num_classes = get_class_names_and_count(train_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=[0.80, 1.20],
    brightness_range=[0.5, 1.5],
    channel_shift_range=25.0,
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
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

def build_mobilenetv2(num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)
    return model

def try_load_model(path, num_classes):
    try:
        loaded_model = load_model(path)
        if loaded_model.output_shape[-1] != num_classes:
            experiment_history.append("Output classes differ. Build new model.")
            return None
        return loaded_model
    except Exception as e:
        experiment_history.append(f"Exception loading previous model: {str(e)}. Build new model.")
        return None

model = try_load_model(best_model_path, num_classes)
if model is None:
    model = build_mobilenetv2(num_classes)
    experiment_history.append("New MobileNetV2 model generated at start.")
else:
    experiment_history.append("Loaded previous best model and will continue training.")

model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

class_counts = train_generator.classes
_, counts = np.unique(class_counts, return_counts=True)
class_weight = dict(enumerate(np.max(counts) / counts))

checkpoint_cb = ModelCheckpoint(
    best_model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=1e-6
)
csvlogger_cb = CSVLogger(history_csv_path, append=True)

callbacks = [checkpoint_cb, reduce_lr_cb, csvlogger_cb]

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight
)

model.save(final_model_path)

if os.path.exists(raw_test_dir) and any(
    f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(raw_test_dir)
):
    result_rows = []
    for fname in os.listdir(raw_test_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(raw_test_dir, fname)
        img = load_img(img_path, target_size=(img_size, img_size))
        x = img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        pred_max = np.max(preds)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        is_unknown = int(pred_max < 0.45)
        pred_label = class_names[pred_class_idx] if (not is_unknown and pred_class_idx < len(class_names)) else 'unknown_or_empty'
        result_rows.append({
            "filename": fname,
            "pred_class": pred_label,
            "pred_prob": float(pred_max),
            "is_unknown_or_empty": is_unknown
        })
    df = pd.DataFrame(result_rows)
    df.to_csv(random_test_csv_path, index=False)
else:
    experiment_history.append("raw_dataset does not exist or contains no images, random evaluation skipped.")

with open(os.path.join(base_dir, 'experiment_history.txt'), 'w') as ef:
    for row in experiment_history:
        ef.write(str(row)+"\n")