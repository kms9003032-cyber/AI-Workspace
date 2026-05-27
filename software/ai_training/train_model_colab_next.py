import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')

img_height = 224
img_width = 224
batch_size = 32
epochs = 100
seed = 42

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.16,
    height_shift_range=0.16,
    shear_range=0.18,
    zoom_range=0.18,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=35,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)
random_test_generator = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

from sklearn.utils.class_weight import compute_class_weight
y_train = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.28)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.42)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csvlogger_cb = CSVLogger('history_colab.csv', append=False)
reducelr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.45,
    patience=7,
    min_lr=1e-6,
    verbose=1
)
callbacks = [checkpoint_cb, csvlogger_cb, reducelr_cb]

steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

model.save('final_model_colab.keras')

from sklearn.metrics import classification_report, confusion_matrix
random_test_generator.reset()
preds = model.predict(random_test_generator, steps=random_test_generator.samples, verbose=1)
y_true = random_test_generator.classes
y_pred = np.argmax(preds, axis=1)
categories = list(random_test_generator.class_indices.keys())
filenames = random_test_generator.filenames
res_df = pd.DataFrame({
    'filename': filenames,
    'true_class': [categories[i] for i in y_true],
    'pred_class': [categories[i] for i in y_pred]
})
for idx, c in enumerate(categories):
    res_df[f'prob_{c}'] = preds[:, idx]
res_df.to_csv('random_test_results.csv', index=False)