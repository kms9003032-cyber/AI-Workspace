import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from collections import Counter
import glob

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
best_model_path = os.path.join(base_dir, 'best_model_colab.keras')
final_model_path = os.path.join(base_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(base_dir, 'history_colab.csv')
random_test_results_csv = os.path.join(base_dir, 'random_test_results.csv')
experiment_history_path = os.path.join(base_dir, 'experiment_history.txt')

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 100

def get_class_weights(generator):
    counter = Counter(generator.classes)
    total = sum(counter.values())
    class_weights = {i: total/(len(counter)*v) for i,v in counter.items()}
    return class_weights

def save_experiment_history(msg):
    with open(experiment_history_path, 'a') as f:
        f.write(msg.strip()+'\n')

train_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=(0.6, 1.4),
    channel_shift_range=40.0,
    shear_range=0.15,
    zoom_range=0.22,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

val_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

class_weights = get_class_weights(train_gen)

must_train_new = False
load_reason = ""
model = None

if os.path.exists(best_model_path):
    try:
        loaded_model = load_model(best_model_path)
        output_shape = loaded_model.output_shape[-1]
        if output_shape == num_classes:
            model = loaded_model
            load_reason = "Resume from best_model_colab.keras"
        else:
            must_train_new = True
            load_reason = f"Number of classes changed: {output_shape} -> {num_classes}"
    except Exception as e:
        must_train_new = True
        load_reason = f"Failed to load existing model: {str(e)}"
else:
    must_train_new = True
    load_reason = "No best_model_colab.keras found"

if must_train_new:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE+(3,))
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    save_experiment_history(f"New model created: {load_reason}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

ckpt_cb = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csv_cb = CSVLogger(history_csv_path, append=True)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=7,
    verbose=1,
    min_lr=1e-6
)
callbacks = [ckpt_cb, csv_cb, reduce_lr_cb]

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples//BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=val_gen.samples//BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

model.save(final_model_path)

if hasattr(history, "history"):
    pd.DataFrame(history.history).to_csv(history_csv_path, index=False)

def get_img_files(folder):
    if not os.path.exists(folder): return []
    exts = ('*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG')
    files = []
    for ext in exts: files.extend(glob.glob(os.path.join(folder, ext)))
    return [f for f in files if os.path.isfile(f) and (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png'))]

if os.path.exists(raw_test_dir):
    raw_img_files = get_img_files(raw_test_dir)
    results = []
    if os.path.exists(best_model_path):
        model_for_eval = load_model(best_model_path)
    else:
        model_for_eval = model
    for fpath in raw_img_files:
        try:
            img = load_img(fpath, target_size=IMAGE_SIZE)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            preds = model_for_eval.predict(arr)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            pred_label = inv_class_indices.get(pred_idx, str(pred_idx))
            filename = os.path.basename(fpath)
            results.append({
                'filename': filename,
                'pred_label_idx': pred_idx,
                'pred_label': pred_label,
                'confidence': confidence
            })
        except Exception as e:
            results.append({
                'filename': os.path.basename(fpath),
                'pred_label_idx': -1,
                'pred_label': 'error',
                'confidence': 0.0
            })
    pd.DataFrame(results).to_csv(random_test_results_csv, index=False)
else:
    pass
