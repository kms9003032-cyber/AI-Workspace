import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')

batch_size = 32
img_size = (224, 224)
epochs = 100
random_seed = 777

class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
n_classes = len(class_names)

def get_class_counts(dir_path, class_names):
    counts = {}
    for c in class_names:
        c_dir = os.path.join(dir_path, c)
        if os.path.exists(c_dir):
            counts[c] = len([f for f in os.listdir(c_dir) if os.path.isfile(os.path.join(c_dir, f))])
        else:
            counts[c] = 0
    return counts

class_counts = get_class_counts(train_dir, class_names)
labels_list = []
for cl in class_names:
    labels_list += [cl] * class_counts[cl]
if sum(class_counts.values()) > 0:
    class_weights = compute_class_weight('balanced', classes=np.array(class_names), y=labels_list)
    class_weights = dict(zip(range(n_classes), class_weights))
else:
    class_weights = None

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.18,
    brightness_range=[0.5, 1.5],
    channel_shift_range=25.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

val_aug = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=random_seed
)

val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

experiment_history = []
model_path = 'best_model_colab.keras'
best_model_exists = os.path.exists(model_path)
retrain_reason = ""
model_loaded = False

if best_model_exists:
    try:
        model = load_model(model_path)
        output_shape = model.layers[-1].output_shape[-1]
        if output_shape != n_classes:
            retrain_reason = f"class_number_mismatch_prev:{output_shape}_now:{n_classes}"
        else:
            model_loaded = True
    except Exception as e:
        retrain_reason = "load_model_failed:" + str(e)

if not model_loaded:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    preds = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    retrain_reason = retrain_reason if retrain_reason else "no_prev_model"
    experiment_history.append({'retrain_reason': retrain_reason, 'n_classes': n_classes})

optimizer = Adam(learning_rate=2e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint('best_model_colab.keras', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=1)
csv_logger_cb = CSVLogger('history_colab.csv', append=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=6, min_lr=1e-6, verbose=1)

callbacks = [checkpoint_cb, csv_logger_cb, reduce_lr_cb]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

model.save('final_model_colab.keras')

def get_image_files(folder):
    if not os.path.exists(folder):
        return []
    valid_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    return [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]

raw_img_files = get_image_files(raw_test_dir)

if len(raw_img_files) > 0:
    raw_pred_results = []
    for fname in raw_img_files:
        try:
            img_path = os.path.join(raw_test_dir, fname)
            img = load_img(img_path, target_size=img_size)
            arr = img_to_array(img) / 255.
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr)
            pred_idx = np.argmax(pred)
            pred_confidence = float(np.max(pred))
            pred_label = class_names[pred_idx]
            raw_pred_results.append({
                'filename': fname,
                'predicted_class': pred_label,
                'confidence': pred_confidence
            })
        except Exception as e:
            raw_pred_results.append({
                'filename': fname,
                'predicted_class': "error",
                'confidence': 0
            })

    df_pred = pd.DataFrame(raw_pred_results)
    df_pred.to_csv('random_test_results.csv', index=False)
else:
    pass