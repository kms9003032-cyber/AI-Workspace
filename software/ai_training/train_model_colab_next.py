import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_dataset')

image_size = (224, 224)
batch_size = 32
epochs = 100
initial_lr = 1e-4

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=(0.7, 1.3),
    shear_range=0.13,
    zoom_range=[0.8, 1.3],
    channel_shift_range=40.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=lambda img: tf.image.random_saturation(img, 0.5, 2.0)
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
idx2class = {v: k for k, v in class_indices.items()}

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=initial_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
callbacks = [checkpoint, csv_logger, reduce_lr]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks
)
model.save('final_model_colab.keras')

pd.DataFrame(history.history).to_csv('history_colab.csv', index=False)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    random_test_dir,
    target_size=image_size,
    batch_size=1,
    class_mode=None,
    shuffle=False
)

best_model = tf.keras.models.load_model('best_model_colab.keras')
preds = best_model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(preds, axis=1)
filenames = test_generator.filenames
results = pd.DataFrame({
    'filename': filenames,
    'predicted_class': [idx2class[i] for i in predicted_classes],
    'prob': np.max(preds, axis=1)
})
results.to_csv('random_test_results.csv', index=False)