import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import classification_report

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
BATCH_SIZE = 24
TARGET_SIZE = (224, 224)
EPOCHS = 100
experiment_history = {'note': []}

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=180,
    width_shift_range=0.17,
    height_shift_range=0.17,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.7, 1.3),
    channel_shift_range=22.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

classes = list(train_generator.class_indices.keys())
class_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in classes]
total = np.sum(class_counts)
class_weights = {i: total / (len(classes) * n) for i, n in enumerate(class_counts)}
for i, c in enumerate(classes):
    if 'unknown' in c or 'empty' in c:
        class_weights[i] = class_weights[i] * 2.0

def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=TARGET_SIZE + (3,), include_top=False, weights='imagenet'
    )
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, output)
    return model

def check_model_compatible(model, num_classes):
    try:
        out_shape = model.output_shape[-1]
        return out_shape == num_classes
    except Exception:
        return False

model_path = 'best_model_colab.keras'
if os.path.exists(model_path):
    try:
        loaded_model = load_model(model_path, compile=False)
        if check_model_compatible(loaded_model, len(classes)):
            loaded_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            model = loaded_model
        else:
            model = build_model(len(classes))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            experiment_history['note'].append('MODEL_REBUILD_DUE_TO_CLASS_MISMATCH')
    except Exception:
        model = build_model(len(classes))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        experiment_history['note'].append('MODEL_REBUILD_LOAD_FAIL')
else:
    model = build_model(len(classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    experiment_history['note'].append('MODEL_INIT_START_TRAIN')

checkpoint_cb = ModelCheckpoint(
    'best_model_colab.keras', monitor='val_loss', save_best_only=True, verbose=1
)
csv_logger = CSVLogger('history_colab.csv')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
callbacks = [checkpoint_cb, csv_logger, reduce_lr]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

model.save('final_model_colab.keras')

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_colab.csv', index=False)

if os.path.exists(raw_test_dir) and len(os.listdir(raw_test_dir)) > 0:
    idx2class = {v: k for k, v in train_generator.class_indices.items()}
    test_images = [f for f in os.listdir(raw_test_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    for fname in test_images:
        path = os.path.join(raw_test_dir, fname)
        try:
            img = tf.keras.utils.load_img(path, target_size=TARGET_SIZE)
            x = tf.keras.utils.img_to_array(img)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x, verbose=0)[0]
            top_idx = np.argmax(preds)
            prob = float(preds[top_idx])
            label = idx2class[top_idx]
            results.append({
                'file': fname, 'predicted_class': label,
                'confidence': prob
            })
        except Exception as e:
            results.append({'file': fname, 'predicted_class': 'error', 'confidence': -1})
    pd.DataFrame(results).to_csv('random_test_results.csv', index=False)
else:
    experiment_history['note'].append('RAW_TESTSET_NOT_FOUND_OR_EMPTY')

if experiment_history['note']:
    pd.DataFrame({'note': experiment_history['note']}).to_csv('experiment_history.csv', index=False)