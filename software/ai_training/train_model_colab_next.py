import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

base_dir = "/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
raw_dataset_dir = os.path.join(base_dir, "raw_dataset")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
DROPOUT_RATE = 0.5
L2_WEIGHT = 1e-4
INIT_LR = 1e-3

def strong_contrast(img):
    img = tf.image.random_contrast(img, lower=0.5, upper=1.7)
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_saturation(img, lower=0.6, upper=1.5)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.5, 1.5],
    shear_range=0.12,
    zoom_range=[0.7, 1.5],
    channel_shift_range=35.0,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=strong_contrast
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

labels = []
for _, y in train_gen:
    labels.extend(np.argmax(y, axis=1))
    if len(labels) >= train_gen.samples:
        break
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=labels)
class_weights = dict(enumerate(class_weights))

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(L2_WEIGHT))(x)
x = Dropout(DROPOUT_RATE)(x)
output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(L2_WEIGHT))(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    "best_model_colab.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    verbose=1
)
csv_logger_cb = CSVLogger("history_colab.csv")
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, csv_logger_cb, reduce_lr_cb]
)

model.save("final_model_colab.keras")

def evaluate_raw_dataset(model, raw_dir, output_csv):
    if not os.path.exists(raw_dir):
        print("raw_dataset 폴더가 없음. 평가 건너뜀.")
        return
    raw_datagen = ImageDataGenerator(rescale=1./255)
    raw_gen = raw_datagen.flow_from_directory(
        raw_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    y_true = raw_gen.classes
    y_pred_prob = model.predict(raw_gen, steps=raw_gen.samples)
    y_pred = np.argmax(y_pred_prob, axis=1)
    filenames = raw_gen.filenames
    class_labels = list(raw_gen.class_indices.keys())
    df = pd.DataFrame({
        'filename': filenames,
        'true_label': [class_labels[idx] for idx in y_true],
        'pred_label': [class_labels[idx] for idx in y_pred],
        'confidence': np.max(y_pred_prob, axis=1)
    })
    df.to_csv(output_csv, index=False)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_csv.replace('.csv', '_report.csv'))
    cm = confusion_matrix(y_true, y_pred)
    np.save(output_csv.replace('.csv', '_cm.npy'), cm)

try:
    evaluate_raw_dataset(model, raw_dataset_dir, "random_test_results.csv")
except Exception as e:
    print("raw_dataset 평가에서 오류 발생:", e)