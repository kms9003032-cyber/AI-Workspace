import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

base_dir = "/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
raw_test_dir = os.path.join(base_dir, "raw_dataset")
img_height, img_width = 224, 224
batch_size = 32
epochs = 100
experiment_history = []

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.18,
    brightness_range=(0.72, 1.28),
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
    class_mode='categorical'
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
idx_to_class = {v: k for k, v in class_indices.items()}
labels = train_generator.classes
try:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))
except Exception as e:
    class_weights = None

model_path = "best_model_colab.keras"
resume_train = False
if os.path.exists(model_path):
    try:
        loaded_model = load_model(model_path)
        if (loaded_model.output_shape[-1] == num_classes):
            model = loaded_model
            resume_train = True
        else:
            experiment_history.append("New model: output_shape does not match, from scratch.")
            base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet")
            base_model.trainable = True
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.3)(x)
            output = Dense(num_classes, activation="softmax")(x)
            model = Model(base_model.input, output)
    except Exception as e:
        experiment_history.append(f"New model: failed to load previous model. {e}")
        base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet")
        base_model.trainable = True
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.3)(x)
        output = Dense(num_classes, activation="softmax")(x)
        model = Model(base_model.input, output)
else:
    experiment_history.append("New model: no existing best_model_colab.keras found.")
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet")
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(base_model.input, output)

optimizer = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint(
    model_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max", save_weights_only=False
)
final_checkpoint = ModelCheckpoint(
    "final_model_colab.keras", monitor=None, verbose=0, save_best_only=False, save_weights_only=False
)
csv_logger = CSVLogger("history_colab.csv", append=resume_train)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7)

callbacks = [checkpoint, final_checkpoint, csv_logger, reduce_lr]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size + int(train_generator.samples % batch_size > 0),
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size + int(val_generator.samples % batch_size > 0),
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

model.save("final_model_colab.keras")

history_df = pd.DataFrame(history.history)
if os.path.exists("history_colab.csv"):
    old_hist = pd.read_csv("history_colab.csv")
    history_df = pd.concat([old_hist, history_df]).reset_index(drop=True)
history_df.to_csv("history_colab.csv", index=False)

def allowed_image_file(filename):
    name = filename.lower()
    return any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png"])

if os.path.isdir(raw_test_dir):
    raw_results = []
    for fname in sorted(os.listdir(raw_test_dir)):
        if not allowed_image_file(fname):
            continue
        img_path = os.path.join(raw_test_dir, fname)
        try:
            img = load_img(img_path, target_size=(img_height, img_width))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)
            pred_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            pred_class = idx_to_class[pred_idx]
            raw_results.append({
                "filename": fname,
                "pred_class": pred_class,
                "confidence": confidence,
                **{f"prob_{idx_to_class[i]}": float(preds[0][i]) for i in range(num_classes)}
            })
        except Exception as e:
            raw_results.append({"filename": fname, "pred_class": "error", "confidence": 0.0, "error": str(e)})
    pd.DataFrame(raw_results).to_csv("random_test_results.csv", index=False)
else:
    experiment_history.append("raw_dataset not found, skipping random test evaluation.")

with open("experiment_history.txt", "w", encoding="utf-8") as f:
    for line in experiment_history:
        f.write(line + "\n")