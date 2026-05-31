import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_dataset_dir = os.path.join(base_dir, 'raw_dataset')
best_model_path = os.path.join(base_dir, 'best_model_colab.keras')
final_model_path = os.path.join(base_dir, 'final_model_colab.keras')
csv_logger_path = os.path.join(base_dir, 'history_colab.csv')
random_test_result_csv = os.path.join(base_dir, 'random_test_results.csv')

img_size = (224, 224)
batch_size = 32
seed = 77

def gripper_style_augmentation(img):
    img = tf.image.random_brightness(img, max_delta=0.27)
    img = tf.image.random_contrast(img, 0.65, 1.35)
    img = tf.image.random_hue(img, 0.035)
    img = tf.image.random_saturation(img, 0.6, 1.4)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_jpeg_quality(img, 65, 100)
    if tf.random.uniform(()) > 0.7:
        img = tf.image.stateless_random_crop(img, size=[int(img_size[0]*0.92), int(img_size[1]*0.92), 3], seed=[seed, seed])
        img = tf.image.resize(img, img_size)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.015)
    img = img + noise
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.16,
    height_shift_range=0.16,
    zoom_range=0.18,
    shear_range=0.09,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.6, 1.5),
    fill_mode='nearest',
    preprocessing_function=gripper_style_augmentation
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

num_classes = len(train_gen.class_indices)
class_indices_rev = {v:k for k,v in train_gen.class_indices.items()}

base_model = MobileNetV2(input_shape=img_size+(3,), weights='imagenet', include_top=False)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.44)(x)
pred = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=pred)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    best_model_path, 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='max'
)
csv_logger = CSVLogger(csv_logger_path, append=False)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=6, 
    min_lr=1e-6, 
    verbose=1
)

epochs = 100
callbacks = [checkpoint, csv_logger, reduce_lr]

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks
)

model.save(final_model_path)

if os.path.exists(raw_dataset_dir) and len(os.listdir(raw_dataset_dir)) > 0:
    test_gen = val_datagen.flow_from_directory(
        raw_dataset_dir,
        target_size=img_size,
        batch_size=1,
        shuffle=False,
        class_mode='categorical'
    )
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    fnames = test_gen.filenames
    y_conf = np.max(y_prob, axis=1)
    df = pd.DataFrame({
        'filename': fnames,
        'true_label': [class_indices_rev[c] for c in y_true],
        'pred_label': [class_indices_rev[c] for c in y_pred],
        'confidence': y_conf
    })
    df.to_csv(random_test_result_csv, index=False)
else:
    with open(random_test_result_csv, 'w') as f:
        f.write('no raw_dataset present\n')
