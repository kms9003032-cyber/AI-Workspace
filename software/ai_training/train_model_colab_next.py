import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight

base_dir = "/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
raw_test_dir = os.path.join(base_dir, "raw_dataset")
best_model_path = os.path.join(base_dir, "best_model_colab.keras")
final_model_path = os.path.join(base_dir, "final_model_colab.keras")
history_csv_path = os.path.join(base_dir, "history_colab.csv")
raw_test_csv_path = os.path.join(base_dir, "random_test_results.csv")
experiment_history_txt = os.path.join(base_dir, "experiment_history.txt")

batch_size = 32
img_height, img_width = 224, 224
epochs = 100
random_seed = 42

tf.random.set_seed(random_seed)
np.random.seed(random_seed)

def get_class_names(directory):
    folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    folders.sort()
    return folders

class_names = get_class_names(train_dir)
num_classes = len(class_names)

def get_class_indices(directory):
    class_names = get_class_names(directory)
    return {name: i for i, name in enumerate(class_names)}

train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=random_seed
)
val_ds = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    seed=random_seed
)

data_augmentation = keras.Sequential([
    layers.RandomRotation(180/360, fill_mode='nearest', seed=random_seed),
    layers.RandomFlip("horizontal_and_vertical", seed=random_seed),
    layers.RandomTranslation(0.1, 0.1, fill_mode='nearest', seed=random_seed),
    layers.RandomZoom(0.2, 0.2, fill_mode='nearest', seed=random_seed),
    layers.RandomContrast(0.2, seed=random_seed),
    layers.RandomBrightness(0.2, seed=random_seed),
    layers.GaussianNoise(0.07, seed=random_seed),
])

def preproc(ds, training):
    if training:
        return ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds

train_ds = preproc(train_ds, True).prefetch(tf.data.AUTOTUNE)
val_ds = preproc(val_ds, False).prefetch(tf.data.AUTOTUNE)

def compute_weights():
    y_labels = []
    temp_ds = image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )
    for _, y in temp_ds:
        y_labels.extend(np.argmax(y.numpy(), axis=1))
    weights_vec = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_labels)
    class_weight = {i: w for i, w in enumerate(weights_vec)}
    return class_weight

if "unknown_or_empty" in class_names:
    class_weight = compute_weights()
else:
    class_weight = None

def same_model_struct(model_path, target_num_classes):
    try:
        m = load_model(model_path)
        if m.layers[-1].output_shape[-1] == target_num_classes:
            del m
            return True
        else:
            del m
            return False
    except Exception:
        return False

model_is_loaded = False
experiment_message = ""
if os.path.exists(best_model_path) and same_model_struct(best_model_path, num_classes):
    try:
        model = load_model(best_model_path)
        model_is_loaded = True
        experiment_message = "Loaded existing best_model_colab.keras for continued training."
    except Exception as e:
        model_is_loaded = False
        experiment_message = f"Failed to load existing model, initializing new model. Error: {str(e)}"
if not model_is_loaded:
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model = keras.Sequential([
        layers.Rescaling(1./255),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    experiment_message = "Created new MobileNetV2 model due to absence/incompatibility of existing model."
try:
    with open(experiment_history_txt, "a") as f:
        f.write(experiment_message + "\n")
except Exception:
    pass

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csv_logger_cb = CSVLogger(history_csv_path)
lr_reduce_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)
callbacks = [checkpoint_cb, csv_logger_cb, lr_reduce_cb]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weight
)

model.save(final_model_path)

def get_image_files(folder):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = []
    if not os.path.isdir(folder):
        return []
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            files.append(os.path.join(folder, f))
    return sorted(files)

def preprocess_raw_img(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    return arr

def predict_raw_dataset(model, raw_test_dir, class_names, csv_path):
    files = get_image_files(raw_test_dir)
    if not files:
        return
    X = np.stack([preprocess_raw_img(f) for f in files])
    preds = model.predict(X, batch_size=batch_size)
    top1_idx = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    result = pd.DataFrame({
        "image_path": files,
        "pred_label": [class_names[i] for i in top1_idx],
        "confidence": confidences
    })
    result.to_csv(csv_path, index=False)

predict_raw_dataset(model, raw_test_dir, class_names, raw_test_csv_path)