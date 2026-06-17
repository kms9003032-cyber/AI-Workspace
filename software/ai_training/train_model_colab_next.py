import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
img_height = 224
img_width = 224
batch_size = 32
epochs = 100
best_model_path = 'best_model_colab.keras'
final_model_path = 'final_model_colab.keras'
history_path = 'history_colab.csv'
random_test_csv = 'random_test_results.csv'

experiment_history = []

def get_class_indices_and_labels(directory):
    gen = ImageDataGenerator().flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    return gen.class_indices, list(gen.class_indices.keys())

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=0.24,
    brightness_range=[0.5, 1.5],
    channel_shift_range=25,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
y_train = train_generator.classes
class_weight_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(zip(np.unique(y_train), class_weight_array))

def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(img_height, img_width,3), include_top=False, weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)
    output = Dense(num_classes, activation='softmax')(x)
    m = Model(inputs=base_model.input, outputs=output)
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

load_success = False
if os.path.exists(best_model_path):
    try:
        loaded_model = load_model(best_model_path)
        loaded_classes = loaded_model.output_shape[-1]
        if loaded_classes == num_classes:
            model = loaded_model
            load_success = True
            experiment_history.append('load_model:success')
        else:
            experiment_history.append(f'load_model:class_mismatch ({loaded_classes} vs {num_classes}), rebuild')
            model = build_model(num_classes)
    except Exception as e:
        experiment_history.append(f'load_model:failed ({str(e)}), rebuild')
        model = build_model(num_classes)
else:
    experiment_history.append('load_model:not_found, new model created')
    model = build_model(num_classes)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.45,
    verbose=1,
    min_lr=3e-6
)
csv_logger = CSVLogger(history_path, append=True)
checkpoint = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
callbacks = [reduce_lr, csv_logger, checkpoint]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save(final_model_path)

if os.path.exists(best_model_path):
    try:
        best_model = load_model(best_model_path)
    except Exception:
        best_model = model
else:
    best_model = model

if os.path.exists(raw_test_dir) and os.path.isdir(raw_test_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(raw_test_dir)
                   if isinstance(f, str) and f.lower().endswith(valid_exts)]
    raw_results = []
    if len(image_files) > 0:
        label_map_inv = {v: k for k, v in class_indices.items()}
        for fname in image_files:
            try:
                img_path = os.path.join(raw_test_dir, fname)
                img = load_img(img_path, target_size=(img_height,img_width))
                arr = img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = best_model.predict(arr)
                pred_label_idx = int(np.argmax(pred[0]))
                pred_label = label_map_inv.get(pred_label_idx, str(pred_label_idx))
                confidence = float(pred[0][pred_label_idx])
                raw_results.append({
                    'filename': fname,
                    'pred_label': pred_label,
                    'confidence': confidence
                })
            except Exception as e:
                raw_results.append({
                    'filename': fname,
                    'pred_label': 'error',
                    'confidence': 0.0
                })
        pd.DataFrame(raw_results).to_csv(random_test_csv, index=False)
    else:
        experiment_history.append('raw_dataset:empty, skipped evaluation')
else:
    experiment_history.append('raw_dataset:not_found, skipped evaluation')

if not os.path.exists(history_path):
    pd.DataFrame(history.history).to_csv(history_path, index=False)

if len(experiment_history) > 0:
    with open('experiment_history.txt', 'a') as f:
        for item in experiment_history:
            f.write(item+'\n')