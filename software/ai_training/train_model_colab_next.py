import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'raw_dataset')

img_height, img_width = 224, 224
batch_size = 32

num_classes = len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.6, 1.4],
    shear_range=0.18,
    zoom_range=0.28,
    channel_shift_range=48,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

base_model = MobileNetV2(include_top=False, input_shape=(img_height, img_width, 3), weights='imagenet')
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.55)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00055),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=5, min_lr=1e-6, verbose=1)

history = model.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=(train_generator.samples // batch_size) + 1,
    validation_data=val_generator,
    validation_steps=(val_generator.samples // batch_size) + 1,
    callbacks=[checkpoint, csv_logger, reduce_lr]
)

model.save('final_model_colab.keras')

raw_generator = val_datagen.flow_from_directory(
    random_test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

y_pred_prob = model.predict(raw_generator, steps=raw_generator.samples, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = raw_generator.classes
label_map = {v: k for k, v in raw_generator.class_indices.items()}
results = []
for idx, (filename, true_idx, pred_idx) in enumerate(zip(raw_generator.filenames, y_true, y_pred)):
    true_label = label_map[true_idx]
    pred_label = label_map[pred_idx]
    confidence = float(np.max(y_pred_prob[idx]))
    results.append({
        'filename': filename,
        'true_label': true_label,
        'pred_label': pred_label,
        'confidence': confidence,
        'correct': int(true_idx == pred_idx)
    })
results_df = pd.DataFrame(results)
results_df.to_csv('random_test_results.csv', index=False)