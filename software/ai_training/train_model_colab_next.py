import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'raw_dataset')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=(0.55, 1.5),
    channel_shift_range=35.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

from collections import Counter
train_classes = train_generator.classes
counter = Counter(train_classes)
max_c = max(counter.values())
class_weight = {i: max_c / c for i, c in counter.items()}

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

optimizer = keras.optimizers.Adam(learning_rate=2e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    verbose=1
)
csv_logger_cb = CSVLogger('history_colab.csv')
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=5e-6
)
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
callbacks = [checkpoint_cb, csv_logger_cb, reduce_lr_cb, earlystop_cb]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

model.save('final_model_colab.keras')

val_hist_df = pd.DataFrame(history.history)
val_hist_df.to_csv('history_colab.csv', index=False)

test_generator = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
pred_probs = model.predict(test_generator, verbose=2)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_generator.classes

from sklearn.metrics import classification_report
target_names = list(test_generator.class_indices.keys())
report = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('random_test_results.csv')