import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
best_model_path = os.path.join(models_dir, 'best_model_colab.keras')
final_model_path = os.path.join(models_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(base_dir, 'history_colab.csv')
raw_eval_csv_path = os.path.join(base_dir, 'random_test_results.csv')
experiment_history_path = os.path.join(base_dir, 'experiment_report.txt')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
initial_lr = 2e-5

def get_class_labels_and_count(directory):
    from glob import glob
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    num_classes = len(class_names)
    return class_names, num_classes

train_labels, num_classes = get_class_labels_and_count(train_dir)
val_labels, val_num_classes = get_class_labels_and_count(val_dir)

assert set(train_labels) == set(val_labels), "Train/val class set mismatch!"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.12,
    shear_range=0.05,
    brightness_range=[0.75, 1.18],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255
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

# class_weights 계산
from sklearn.utils.class_weight import compute_class_weight
labels = train_generator.classes
class_weights_ = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights_))

# 모델 이어학습 혹은 최초생성
def load_or_build_model():
    if os.path.exists(best_model_path):
        try:
            loaded_model = load_model(best_model_path, compile=False)
            if loaded_model.output_shape[-1] == num_classes:
                print("Continue training from previous best_model_colab.keras")
                start_acc = None
                if os.path.exists(history_csv_path):
                    try:
                        hist_df = pd.read_csv(history_csv_path)
                        start_acc = hist_df['val_accuracy'].iloc[-1]
                    except Exception: pass
                experiment_text = f'Continue train: loaded best_model_colab.keras, out_classes={num_classes}, prev_best_val_acc={start_acc}\n'
                with open(experiment_history_path, 'a') as f:
                    f.write(experiment_text)
                return loaded_model, True
        except Exception as e:
            experiment_text = f'Could not load previous model: {str(e)}. Starting new model.\n'
            with open(experiment_history_path, 'a') as f:
                f.write(experiment_text)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.18)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.15)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, out)
    experiment_text = f'New model built. out_classes = {num_classes}\n'
    with open(experiment_history_path, 'a') as f:
        f.write(experiment_text)
    return model, False

model, resumed = load_or_build_model()
optimizer = Adam(learning_rate=initial_lr)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.06),
    metrics=['accuracy']
)

mc = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=6,
    min_lr=5e-6,
    verbose=1
)
csv_logger = CSVLogger(history_csv_path, append=True if resumed else False)

callbacks = [mc, reduce_lr, csv_logger]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

model.save(final_model_path)

# load best for raw_dataset 평가
if os.path.exists(best_model_path):
    best_model = load_model(best_model_path)
else:
    best_model = model

if os.path.exists(raw_test_dir):
    raw_files = [
        os.path.join(raw_test_dir, fname)
        for fname in os.listdir(raw_test_dir)
        if any(ext in fname.lower() for ext in ['jpg','jpeg','png'])
    ]
    results = []
    class_indices = train_generator.class_indices.copy()
    inv_class_indices = {v:k for k,v in class_indices.items()}
    for img_path in raw_files:
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            arr = img_to_array(img)
            arr = arr / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = best_model.predict(arr, verbose=0)
            pred_idx = np.argmax(pred)
            pred_label = inv_class_indices[pred_idx]
            confidence = float(np.max(pred))
        except Exception as e:
            pred_label = 'error'
            confidence = 0.0
        # True label 판단 (filename에 class와 일치하는 경우만)
        fname = os.path.basename(img_path)
        matched_true_label = None
        for c in class_indices:
            if c in fname:
                matched_true_label = c
                break
        is_correct = (matched_true_label == pred_label) if matched_true_label is not None else None
        results.append({
            'filename': fname,
            'pred_label': pred_label,
            'confidence': confidence,
            'true_label': matched_true_label,
            'is_correct': is_correct
        })
    df = pd.DataFrame(results)
    df.to_csv(raw_eval_csv_path, index=False)
    corrects = [r['is_correct'] for r in results if r['is_correct'] is not None]
    accuracy = np.mean(corrects) if corrects else None
    experiment_line = f'raw_dataset evaluated on {len(raw_files)} images, accuracy={accuracy}\n'
    with open(experiment_history_path, 'a') as f:
        f.write(experiment_line)
else:
    with open(experiment_history_path, 'a') as f:
        f.write('raw_dataset folder not found, skipping raw_dataset evaluation\n')