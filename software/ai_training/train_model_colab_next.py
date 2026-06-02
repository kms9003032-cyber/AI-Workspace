import os
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
models_dir = os.path.join(base_dir, 'models')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
best_model_path = os.path.join(models_dir, 'best_model_colab.keras')
final_model_path = os.path.join(models_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(models_dir, 'history_colab.csv')
random_test_result_csv = os.path.join(models_dir, 'random_test_results.csv')
img_size = (224, 224)
batch_size = 32
epochs = 100

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def valid_image_file(filename):
    lower = filename.lower()
    return ('.jpg' in lower or '.jpeg' in lower or '.png' in lower) and not lower.startswith('.')

experiment_report = []
initial_val_acc = None

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.08,
    shear_range=0.02,
    brightness_range=(0.85, 1.15),
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
inverse_class_indices = {v: k for k, v in class_indices.items()}

model = None
is_initial_training = True
old_val_acc = None
fine_tune_start = 0

try:
    if os.path.exists(best_model_path):
        loaded_model = load_model(best_model_path)
        loaded_output_shape = loaded_model.output.shape[-1]
        if loaded_output_shape == num_classes:
            model = loaded_model
            is_initial_training = False
            experiment_report.append(f'[{datetime.now()}] Continue training from best_model_colab.keras.')
            eval_result = model.evaluate(val_gen, verbose=0)
            initial_val_acc = eval_result[1]
            experiment_report.append(f'Previous best model val_accuracy: {initial_val_acc:.4f}')
            lr = 1e-5
        else:
            experiment_report.append(f'[{datetime.now()}] Existing model output shape ({loaded_output_shape}) != current class count ({num_classes}), start fresh.')
            is_initial_training = True
except Exception as e:
    experiment_report.append(f'[{datetime.now()}] Could not load previous model: {e}, start fresh.')

if model is None:
    base_model = MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.25)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    lr = 2e-5
    experiment_report.append(f'[{datetime.now()}] Model freshly created with classes: {num_classes}')

callbacks = [
    ModelCheckpoint(
        best_model_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False,
        verbose=1
    ),
    CSVLogger(history_csv_path, append=not is_initial_training),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

loss_fn = CategoricalCrossentropy(label_smoothing=0.05)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=loss_fn,
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    initial_epoch=0 if is_initial_training else None
)

model.save(final_model_path)

val_acc_history = history.history.get('val_accuracy', [])
if len(val_acc_history) >= 1:
    best_epoch_val_acc = float(np.max(val_acc_history))
    experiment_report.append(f'Best val_accuracy during training: {best_epoch_val_acc:.4f}')
    if initial_val_acc is not None:
        experiment_report.append(f'Improvement from previous best_val_accuracy: {best_epoch_val_acc - initial_val_acc:+.4f}')
else:
    experiment_report.append('val_accuracy history missing.')

def get_label_from_filename(filename, classes):
    lower = filename.lower()
    matches = [cls for cls in classes if cls in lower]
    if matches:
        return matches[0]
    return None

raw_eval_results = []
raw_eval_acc = None

if os.path.exists(raw_test_dir):
    raw_files = [f for f in os.listdir(raw_test_dir) if valid_image_file(f)]
    if len(raw_files) > 0:
        y_true = []
        y_pred = []
        y_conf = []
        for fname in raw_files:
            try:
                img_path = os.path.join(raw_test_dir, fname)
                img = load_img(img_path, target_size=img_size)
                x = img_to_array(img) / 255.
                x = np.expand_dims(x, axis=0)
                preds = model.predict(x, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                pred_label = inverse_class_indices[pred_idx]
                pred_conf = float(preds[pred_idx])
                true_label = get_label_from_filename(fname, class_indices.keys())
                is_correct = 1 if (true_label is not None and pred_label == true_label) else 0
                raw_eval_results.append({
                    'filename': fname,
                    'true_label': true_label if true_label is not None else '',
                    'pred_label': pred_label,
                    'confidence': pred_conf,
                    'is_correct': is_correct
                })
                if true_label is not None:
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    y_conf.append(pred_conf)
            except Exception as e:
                continue
        if len([r for r in raw_eval_results if r['true_label'] != '']) > 0:
            correct = sum(1 for r in raw_eval_results if (r['true_label'] != '' and r['pred_label'] == r['true_label']))
            total = sum(1 for r in raw_eval_results if (r['true_label'] != ''))
            raw_eval_acc = correct / total if total > 0 else None
            experiment_report.append(f'raw_dataset accuracy (w/ true label in filename): {raw_eval_acc:.4f}')
        pd.DataFrame(raw_eval_results).to_csv(random_test_result_csv, index=False)
    else:
        experiment_report.append('raw_dataset exists but contains no valid images.')
else:
    experiment_report.append('raw_dataset folder does not exist; skipping raw environment evaluation.')

with open(os.path.join(models_dir, 'experiment_report.txt'), 'a') as report_file:
    for line in experiment_report:
        report_file.write(line + '\n')