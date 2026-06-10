import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42

def get_classes(directory):
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

train_classes = get_classes(train_dir)
num_classes = len(train_classes)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.13,
    height_shift_range=0.13,
    shear_range=0.12,
    zoom_range=0.18,
    brightness_range=[0.65,1.35],
    channel_shift_range=25.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=train_classes,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=train_classes,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

best_model_path = 'best_model_colab.keras'
final_model_path = 'final_model_colab.keras'
csv_log_path = 'history_colab.csv'
raw_test_results_path = 'random_test_results.csv'
experiment_history_path = 'experiment_history.txt'
resume_training = False
experiment_reason = ''
model = None
try:
    if os.path.exists(best_model_path):
        loaded_model = load_model(best_model_path)
        out_shape = loaded_model.output_shape[-1]
        if out_shape == num_classes:
            model = loaded_model
            resume_training = True
        else:
            experiment_reason = '클래스 수 변경으로 인해 모델 구조 다름, 새 모델 생성'
    else:
        experiment_reason = '기존 best_model_colab.keras 없음, 새 모델 생성'
except Exception as e:
    experiment_reason = f'모델 로드 실패: {str(e)} 새 모델 생성'
if not resume_training:
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.18),
        Dense(num_classes, activation='softmax')
    ])
    experiment_reason = experiment_reason or '새 모델 구조, 신규 학습'

if not resume_training and experiment_reason:
    with open(experiment_history_path, 'a') as ef:
        ef.write(experiment_reason+'\n')

optimizer = tf.keras.optimizers.Adam(learning_rate=1.5e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
csvlogger = CSVLogger(csv_log_path, append=resume_training)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.45,
    patience=7,
    min_lr=2e-6,
    verbose=1
)
callbacks = [checkpoint, csvlogger, reduce_lr]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(final_model_path)

if os.path.isdir(raw_test_dir):
    img_files = []
    for fname in os.listdir(raw_test_dir):
        if fname.lower().split('.')[-1] in {'jpg', 'jpeg', 'png'}:
            img_files.append(fname)
    if img_files:
        from tensorflow.keras.utils import img_to_array, load_img
        results = []
        for fname in img_files:
            try:
                img_path = os.path.join(raw_test_dir, fname)
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                x = img_to_array(img)/255.0
                x = np.expand_dims(x, axis=0)
                pred = model.predict(x)
                pred_label = np.argmax(pred[0])
                confidence = float(np.max(pred[0]))
                label_name = train_classes[pred_label]
                results.append({'filename': fname, 'pred_label': label_name, 'confidence': confidence})
            except Exception as e:
                results.append({'filename': fname, 'pred_label': 'error', 'confidence': -1})
        results_df = pd.DataFrame(results)
        results_df.to_csv(raw_test_results_path, index=False)
else:
    pass
