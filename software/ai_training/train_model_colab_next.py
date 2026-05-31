import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import pandas as pd

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=[0.8, 1.2],
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=25.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    seed=SEED,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    seed=SEED,
    class_mode='categorical',
    shuffle=False
)

n_classes = len(train_gen.class_indices)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.025))(x)
x = Dropout(0.5)(x)
output = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = os.path.join(base_dir, 'best_model_colab.keras')
final_path = os.path.join(base_dir, 'final_model_colab.keras')
csvlogger_path = os.path.join(base_dir, 'history_colab.csv')
test_result_path = os.path.join(base_dir, 'random_test_results.csv')

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csv_logger = CSVLogger(csvlogger_path)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=6,
    min_lr=1e-6,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

epochs = 100

history = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=train_gen.samples // batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size,
    callbacks=[checkpoint, csv_logger, reduce_lr, early_stop],
    verbose=1
)

model.save(final_path)

# -----------------------------------------------------------------------------
# RAW DATASET 평가 (폴더 없을 시 평가만 스킵, 전체 코드 실패 방지)
# -----------------------------------------------------------------------------
if os.path.isdir(raw_test_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        base_dir,
        classes=['raw_dataset'],
        target_size=img_size,
        batch_size=1,
        shuffle=False,
        class_mode=None
    )

    preds = model.predict(test_gen, verbose=1)
    pred_scores = np.max(preds, axis=1)
    pred_labels = np.argmax(preds, axis=1)
    idx2class = dict((v, k) for k, v in train_gen.class_indices.items())

    threshold = 0.7
    unknown_idx = train_gen.class_indices.get('unknown_or_empty', None)

    results = []
    for i in range(len(test_gen.filenames)):
        fn = test_gen.filenames[i]
        plabel = pred_labels[i]
        conf = pred_scores[i]
        result_idx = plabel if conf >= threshold else (unknown_idx if unknown_idx is not None else plabel)
        result_class = idx2class[result_idx] if result_idx in idx2class else 'unknown_or_empty'
        results.append({'filename': fn, 'predicted': result_class, 'confidence': float(conf)})

    pd.DataFrame(results).to_csv(test_result_path, index=False)
