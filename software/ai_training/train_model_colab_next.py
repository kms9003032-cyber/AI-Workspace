import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
raw_test_dir = os.path.join(base_dir, "raw_dataset")
best_model_path = os.path.join(base_dir, "best_model_colab.keras")
final_model_path = os.path.join(base_dir, "final_model_colab.keras")
csv_logger_path = os.path.join(base_dir, "history_colab.csv")
random_test_results_path = os.path.join(base_dir, "random_test_results.csv")
img_size = 224
batch_size = 32
epochs = 100
seed = 1212

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.18,
    shear_range=0.12,
    brightness_range=[0.75, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=18.0,
    fill_mode='nearest'
)
val_aug = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_aug.flow_from_directory(
    train_dir, target_size=(img_size,img_size),
    batch_size=batch_size, class_mode='categorical', shuffle=True, seed=seed
)
val_gen = val_aug.flow_from_directory(
    val_dir, target_size=(img_size,img_size),
    batch_size=batch_size, class_mode='categorical', shuffle=False
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
labels = []
for class_folder, i in class_indices.items():
    folder_path = os.path.join(train_dir, class_folder)
    for name in os.listdir(folder_path):
        if name.lower().endswith(('.jpg','.jpeg','.png')):
            labels.append(i)
class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
class_weights = dict(enumerate(class_weights))

use_previous_model = False
experiment_history = {}
if os.path.exists(best_model_path):
    try:
        prev_model = load_model(best_model_path, compile=False)
        prev_layer = prev_model.layers[-1]
        prev_units = getattr(prev_layer, 'units', None)
        if prev_units == num_classes:
            model = prev_model
            use_previous_model = True
            experiment_history['resume'] = True
        else:
            experiment_history['resume'] = False
    except Exception as e:
        use_previous_model = False
        experiment_history['resume_load_model_error'] = str(e)
if not use_previous_model:
    base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    experiment_history['resume'] = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1, save_format='keras', mode='max'),
    ModelCheckpoint(final_model_path, monitor=None, save_best_only=False, verbose=0, save_format='keras', mode='max'),
    CSVLogger(csv_logger_path, append=use_previous_model),
    ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights
)

model.save(final_model_path)

if os.path.exists(best_model_path):
    try:
        best_model = load_model(best_model_path, compile=False)
        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    except Exception as e:
        best_model = model
else:
    best_model = model

if os.path.exists(raw_test_dir) and os.path.isdir(raw_test_dir):
    img_files = []
    for fname in os.listdir(raw_test_dir):
        lname = fname.lower()
        if lname.endswith('.jpg') or lname.endswith('.jpeg') or lname.endswith('.png'):
            img_files.append(fname)
    results = []
    for fname in img_files:
        path = os.path.join(raw_test_dir, fname)
        try:
            img = load_img(path, target_size=(img_size, img_size))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = best_model.predict(arr, verbose=0)[0]
            top_idx = np.argmax(pred)
            confidence = float(pred[top_idx])
            result = {
                "filename": fname,
                "pred_class": list(class_indices.keys())[top_idx],
                "pred_confidence": confidence
            }
            for class_name, idx in class_indices.items():
                result[f"score_{class_name}"] = float(pred[idx])
            results.append(result)
        except Exception as e:
            results.append({
                "filename": fname,
                "pred_class": "ERROR",
                "pred_confidence": 0.0
            })
    df_result = pd.DataFrame(results)
    df_result.to_csv(random_test_results_path, index=False)