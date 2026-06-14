import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from datetime import datetime
import shutil

base_dir = "/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
raw_test_dir = os.path.join(base_dir, "raw_dataset")

image_size = 224
batch_size = 32
epochs = 100
init_lr = 1e-4
seed = 83

experiment_history = []
result_csv_name = os.path.join(base_dir, "random_test_results.csv")
history_csv_name = os.path.join(base_dir, "history_colab.csv")
best_model_path = os.path.join(base_dir, "best_model_colab.keras")
final_model_path = os.path.join(base_dir, "final_model_colab.keras")

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=seed
)
val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=seed
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

model = None
load_success = False
if os.path.exists(best_model_path):
    try:
        prev_model = load_model(best_model_path)
        out_shape = prev_model.layers[-1].output.shape[-1]
        if int(out_shape) == num_classes:
            model = prev_model
            load_success = True
            experiment_history.append(f'[{datetime.now()}] best_model_colab.keras loaded; continue training.')
        else:
            experiment_history.append(f'[{datetime.now()}] best_model_colab.keras loaded but output shape mismatch. Reinitialize model.')
    except Exception as e:
        experiment_history.append(f'[{datetime.now()}] best_model_colab.keras load failed: {e}; Reinitialize model.')

if not load_success:
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(192, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    pred = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=pred)
    experiment_history.append(f'[{datetime.now()}] New MobileNetV2 model (trainable=True) initialized.')

model.compile(
    optimizer=Adam(init_lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

csv_logger = CSVLogger(history_csv_name, append=load_success)
model_ckpt = ModelCheckpoint(
    best_model_path, monitor="val_loss", save_best_only=True, save_weights_only=False, mode="min", verbose=1
)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.33, patience=5, min_lr=3e-6, verbose=1)

callbacks = [csv_logger, model_ckpt, lr_scheduler]

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size,
    callbacks=callbacks,
    verbose=1
)

model.save(final_model_path)

def is_image(filename):
    ext = filename.lower().split('.')[-1]
    return ext in ['jpg','jpeg','png']

def load_images(image_dir):
    files = []
    if os.path.exists(image_dir):
        for fname in os.listdir(image_dir):
            if is_image(fname):
                files.append(fname)
    return files

def preprocess_img(img_path):
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype("float32") / 255.
    return img

try:
    raw_imgs_in_folder = load_images(raw_test_dir)
    if len(raw_imgs_in_folder) > 0:
        X_raw = []
        fname_list = []
        for fname in raw_imgs_in_folder:
            full_path = os.path.join(raw_test_dir, fname)
            img = preprocess_img(full_path)
            if img is not None:
                X_raw.append(img)
                fname_list.append(fname)
            else:
                experiment_history.append(f'[{datetime.now()}] Unable to read image: {fname}')
        if X_raw:
            X_raw_arr = np.stack(X_raw)
            y_preds = model.predict(X_raw_arr, batch_size=8)
            pred_classes_idx = np.argmax(y_preds, axis=1)
            pred_classes = [inv_class_indices.get(idx, "unknown") for idx in pred_classes_idx]
            pred_confidence = np.max(y_preds, axis=1)
            df = pd.DataFrame({'filename': fname_list, 'pred_class': pred_classes, 'confidence': pred_confidence})
            df.to_csv(result_csv_name, index=False)
    else:
        experiment_history.append(f'[{datetime.now()}] No image files in raw_dataset; skipping raw_dataset evaluation.')
except Exception as e:
    experiment_history.append(f'[{datetime.now()}] raw_dataset evaluation error: {e}')

hist_exp_path = os.path.join(base_dir, "experiment_log.txt")
with open(hist_exp_path, "a") as f:
    for row in experiment_history:
        f.write(row + "\n")

with open(os.path.join(base_dir, "class_indices.txt"), "w") as f:
    for k, v in class_indices.items():
        f.write(f"{k},{v}\n")