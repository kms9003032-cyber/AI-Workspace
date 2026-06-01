import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, 'best_model_colab.keras')
final_model_path = os.path.join(model_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(model_dir, 'history_colab.csv')
raw_test_csv_path = os.path.join(model_dir, 'random_test_results.csv')
img_size = 224
batch_size = 32
epochs = 100
seed = 202

classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(classes)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.07,
    height_shift_range=0.07,
    zoom_range=0.10,
    brightness_range=(0.75, 1.25),
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

y_train_labels = []
for cls, idx in train_gen.class_indices.items():
    dir_path = os.path.join(train_dir, cls)
    n = len([f for f in os.listdir(dir_path) if (".jpg" in f.lower() or ".jpeg" in f.lower() or ".png" in f.lower())])
    y_train_labels.extend([idx]*n)
class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_labels)
class_weight = dict(enumerate(class_weights_arr))

initial_lr = 1e-5
finetune_flag = False
continue_train = False
experiment_report = []

if os.path.exists(best_model_path):
    try:
        loaded_model = load_model(best_model_path)
        out_shape = loaded_model.output_shape[-1]
        if out_shape == num_classes:
            model = loaded_model
            for layer in model.layers:
                layer.trainable = True
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            val_steps = val_gen.samples // val_gen.batch_size + (val_gen.samples % val_gen.batch_size > 0)
            prev_eval = model.evaluate(val_gen, steps=val_steps, verbose=0)
            prev_val_acc = float(prev_eval[1])
            experiment_report.append(f'Best model found. Continue training with initial val_accuracy={prev_val_acc:.4f}')
            continue_train = True
        else:
            experiment_report.append(f'Output class mismatch: model={out_shape}, data={num_classes}. Starting new model.')
            continue_train = False
    except Exception as e:
        experiment_report.append(f'Error loading best_model_colab.keras: {e}. Starting new model.')
        continue_train = False
if not continue_train:
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
    base_model.trainable = True
    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    experiment_report.append('Started new MobileNetV2 model. base_model.trainable=True.')

csv_logger = CSVLogger(history_csv_path, append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=1e-6, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint(
    best_model_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode='min'
)
callbacks = [model_checkpoint, csv_logger, reduce_lr]

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weight
)

model.save(final_model_path)

try:
    best_model = load_model(best_model_path)
except Exception as e:
    best_model = model
    experiment_report.append(f'Failed to load best_model for raw_dataset eval, using latest model: {e}')

if os.path.exists(raw_test_dir) and os.path.isdir(raw_test_dir):
    raw_test_files = [f for f in os.listdir(raw_test_dir) if (".jpg" in f.lower() or ".jpeg" in f.lower() or ".png" in f.lower())]
    img_shape = (img_size, img_size)
    raw_results = []
    if len(raw_test_files) > 0:
        class_names = {v: k for k, v in train_gen.class_indices.items()}
        for fname in raw_test_files:
            fpath = os.path.join(raw_test_dir, fname)
            try:
                img = load_img(fpath, target_size=img_shape)
                arr = img_to_array(img) / 255.
                arr = np.expand_dims(arr, axis=0)
                preds = best_model.predict(arr, verbose=0)
                pred_idx = int(np.argmax(preds))
                conf = float(np.max(preds))
                pred_class = class_names[pred_idx]
                true_class = None
                for name in classes:
                    if name in fname:
                        true_class = name
                        break
                is_correct = None
                if true_class is not None:
                    is_correct = int(true_class == pred_class)
                raw_results.append({
                    'filename': fname,
                    'true_class': true_class if true_class is not None else '',
                    'pred_class': pred_class,
                    'confidence': conf,
                    'is_correct': is_correct
                })
            except Exception as e:
                raw_results.append({
                    'filename': fname,
                    'true_class': '',
                    'pred_class': 'error',
                    'confidence': 0.0,
                    'is_correct': None
                })
        pd.DataFrame(raw_results).to_csv(raw_test_csv_path, index=False)
        if any(r['is_correct'] is not None for r in raw_results):
            valid = [r for r in raw_results if r['is_correct'] is not None]
            raw_acc = np.mean([r['is_correct'] for r in valid]) if len(valid) > 0 else 0
            experiment_report.append(f'raw_dataset 실제 평가 이미지: {len(raw_results)}, 정확도(정답 라벨 추출 기준): {raw_acc:.4f}')
    else:
        experiment_report.append('raw_dataset 폴더에는 평가할 이미지가 없음.')
else:
    experiment_report.append('raw_dataset 폴더가 없어서 평가를 생략함.')

with open(os.path.join(model_dir, 'experiment_report.txt'), 'w') as f:
    for line in experiment_report:
        f.write(line + '\n')