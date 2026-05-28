import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

base_dir = "/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
random_test_dir = os.path.join(base_dir, "random_test")
img_size = (224, 224)
batch_size = 32
epochs = 30
init_lr = 1e-3

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.10,
    height_shift_range=0.10,
    brightness_range=[0.5, 1.5],
    shear_range=0.12,
    zoom_range=0.18,
    channel_shift_range=28.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)
val_aug = ImageDataGenerator(rescale=1./255)
train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)
val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)
class_labels = list(train_gen.class_indices.keys())
n_classes = len(class_labels)
labels = train_gen.classes
cw = compute_class_weight(
    "balanced",
    classes=np.arange(n_classes),
    y=labels
)
class_weights = {i: w for i, w in enumerate(cw)}
if "unknown_or_empty" in train_gen.class_indices:
    unknown_idx = train_gen.class_indices["unknown_or_empty"]
    class_weights[unknown_idx] = class_weights[unknown_idx] * 1.7

base_model = MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
out = Dense(n_classes, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
model = Model(base_model.input, out)
model.compile(
    optimizer=Adam(learning_rate=init_lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_best = ModelCheckpoint(
    os.path.join(base_dir, "best_model_colab.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
checkpoint_final = ModelCheckpoint(
    os.path.join(base_dir, "final_model_colab.keras"),
    save_best_only=False,
    save_freq="epoch",
    verbose=0
)
csv_logger = CSVLogger(os.path.join(base_dir, "history_colab.csv"))
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
callbacks = [checkpoint_best, checkpoint_final, csv_logger, reduce_lr, early_stop]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

model = load_model(os.path.join(base_dir, "best_model_colab.keras"))

random_gen = val_aug.flow_from_directory(
    random_test_dir,
    target_size=img_size,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)
steps = random_gen.samples
preds = model.predict(random_gen, steps=steps, verbose=1)
pred_labels = np.argmax(preds, axis=1)
true_labels = random_gen.classes
report = classification_report(
    true_labels, pred_labels, target_names=random_gen.class_indices.keys(), output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(base_dir, "random_test_results.csv"))