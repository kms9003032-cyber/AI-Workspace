import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 100
SEED = 42

def gripper_env_aug(img):
    img = tf.image.random_jpeg_quality(img, 85, 100)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    if tf.random.uniform(()) > 0.7:
        img = tf.image.random_hue(img, max_delta=0.1)
    if tf.random.uniform(()) > 0.6:
        img = tf.image.stateless_random_crop(img, size=(np.random.randint(180,224), np.random.randint(180,224), 3), seed=[SEED, 0])
        img = tf.image.resize(img, IMG_SIZE)
    if tf.random.uniform(()) > 0.5:
        sigma = tf.random.uniform((), 0.5, 1.5)
        shape = tf.concat([IMG_SIZE, [3]], 0)
        noise = tf.random.normal(shape, mean=0.0, stddev=sigma, dtype=tf.float32)
        img = img + noise
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.18,
    height_shift_range=0.18,
    shear_range=0.18,
    zoom_range=0.20,
    brightness_range=(0.55, 1.6),
    channel_shift_range=40.0,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=gripper_env_aug
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
random_test_generator = test_datagen.flow_from_directory(
    random_test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
n_classes = train_generator.num_classes

class_counts = np.bincount(train_generator.classes)
max_count = np.max(class_counts)
class_weights = {i: max_count / (count if count > 0 else 1) for i, count in enumerate(class_counts)}

base_model = MobileNetV2(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights='imagenet'
)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(192, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)
output = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

optimizer = keras.optimizers.Adam(learning_rate=2e-4)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = os.path.join(base_dir, 'best_model_colab.keras')
final_model_path = os.path.join(base_dir, 'final_model_colab.keras')
log_path = os.path.join(base_dir, 'history_colab.csv')
random_result_path = os.path.join(base_dir, 'random_test_results.csv')

os.makedirs(base_dir, exist_ok=True)
ckpt = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode='max'
)
logger = CSVLogger(log_path)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=2e-6,
    verbose=1
)
callbacks = [ckpt, logger, reduce_lr]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

model.save(final_model_path)

model_best = keras.models.load_model(checkpoint_path)
random_test_generator.reset()
preds = model_best.predict(random_test_generator, steps=len(random_test_generator), verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = random_test_generator.classes
labels = list(random_test_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
acc = accuracy_score(y_true, y_pred)
unknown_rate = 0.0
if 'unknown_or_empty' in labels:
    unknown_idx = random_test_generator.class_indices['unknown_or_empty']
    unknown_rate = (y_pred == unknown_idx).sum() / float(len(y_pred))
else:
    unknown_rate = None

result_dict = {
    'test_accuracy': [acc],
    'unknown_or_empty_rate': [unknown_rate if unknown_rate is not None else 'N/A']
}
for cname in labels:
    clsdict = report.get(cname, {})
    result_dict[f'precision_{cname}'] = [clsdict.get('precision', 0.0)]
    result_dict[f'recall_{cname}'] = [clsdict.get('recall', 0.0)]
    result_dict[f'f1_{cname}'] = [clsdict.get('f1-score', 0.0)]
pd.DataFrame(result_dict).to_csv(random_result_path, index=False)