import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_test')

img_height = 224
img_width = 224
batch_size = 32
epochs = 100

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.18,
    brightness_range=(0.5,1.5),
    channel_shift_range=20.,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_generator.class_indices
reverse_class_indices = {v:k for k,v in class_indices.items()}
num_classes = len(class_indices)

base_model = MobileNetV2(
    input_shape=(img_height,img_width,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True

fine_tune_at = len(base_model.layers) // 2
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0005))(x)
model = Model(inputs=base_model.input, outputs=outputs)

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint(
    'best_model_colab.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=3,
    verbose=1,
    min_lr=1e-6
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks=[checkpoint_cb, csv_logger, reduce_lr, early_stop]

steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = val_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)

model.save('final_model_colab.keras')

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
test_generator = test_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_height,img_width),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

test_filenames = test_generator.filenames
test_probs = model.predict(test_generator, steps=len(test_generator), verbose=1)
pred_indices = np.argmax(test_probs, axis=1)
confidences = np.max(test_probs, axis=1)
unknown_threshold = 0.5
pred_labels = []
for i, conf in enumerate(confidences):
    if conf < unknown_threshold:
        pred_labels.append('unknown_or_empty')
    else:
        pred_labels.append(reverse_class_indices[pred_indices[i]])

test_results = pd.DataFrame({
    'filename': test_filenames,
    'predicted_label': pred_labels,
    'confidence': confidences
})
test_results.to_csv('random_test_results.csv', index=False)