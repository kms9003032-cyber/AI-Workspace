import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, 'best_model_colab.keras')
final_model_path = os.path.join(model_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(model_dir, 'history_colab.csv')
random_test_csv_path = os.path.join(model_dir, 'random_test_results.csv')
experiment_report_path = os.path.join(model_dir, 'experiment_report_log.csv')

batch_size = 32
img_size = (224, 224)
epochs = 100
dropout_rate = 0.25

experiment_history = []
initial_best_acc = None
start_acc = None

datagen_kwargs = dict(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=[0.8, 1.15],
    shear_range=0.04,
    fill_mode='nearest'
)

train_datagen = ImageDataGenerator(**datagen_kwargs)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_generator.class_indices
idx2class = {v: k for k, v in class_indices.items()}
num_classes = len(class_indices)

model_loaded = False
initial_weights_loaded = False
if os.path.exists(best_model_path):
    try:
        model = load_model(best_model_path)
        if model.output_shape[-1] == num_classes:
            model_loaded = True
            print('[INFO] Loaded existing best_model_colab.keras for continue training.')
        else:
            model_loaded = False
            experiment_history.append({'reason': 'Class count mismatch. Model re-initialized.'})
    except Exception as e:
        model_loaded = False
        experiment_history.append({'reason': f'Failed to load best model: {e}. Re-initialized.'})

if not model_loaded:
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    experiment_history.append({'reason': 'Initialized new MobileNetV2 model.'})

learning_rate = 2e-5 if model_loaded else 1e-4
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger(history_csv_path)
checkpoint = ModelCheckpoint(
    best_model_path, monitor='val_accuracy', mode='max', save_best_only=True,
    verbose=1, save_weights_only=False
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

start_acc = None
if os.path.exists(history_csv_path) and model_loaded:
    try:
        prior_history = pd.read_csv(history_csv_path)
        if 'val_accuracy' in prior_history.columns:
            initial_best_acc = prior_history['val_accuracy'].max()
        else:
            initial_best_acc = None
    except Exception:
        initial_best_acc = None

val_eval = model.evaluate(val_generator, verbose=0)
if isinstance(val_eval, (list, tuple)) and len(val_eval) > 1:
    start_acc = float(val_eval[1])
else:
    start_acc = None
experiment_history.append({'initial_best_val_accuracy': initial_best_acc, 'starting_val_accuracy': start_acc})

callbacks = [checkpoint, csv_logger, reduce_lr]

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)
pd.DataFrame(history.history).to_csv(history_csv_path, index=False)
model.save(final_model_path)

raw_results = []
raw_acc = None
if os.path.exists(raw_test_dir):
    raw_filenames = [
        f for f in os.listdir(raw_test_dir)
        if ('.jpg' in f.lower() or '.jpeg' in f.lower() or '.png' in f.lower())
           and os.path.isfile(os.path.join(raw_test_dir, f))
    ]
    if len(raw_filenames) > 0:
        for fname in raw_filenames:
            path = os.path.join(raw_test_dir, fname)
            try:
                img = load_img(path, target_size=img_size)
                arr = img_to_array(img).astype('float32') / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr, verbose=0)
                pred_label_idx = np.argmax(pred[0])
                pred_label = idx2class[pred_label_idx]
                true_label = None
                for cname in class_indices.keys():
                    if cname in fname:
                        true_label = cname
                        break
                is_correct = (true_label == pred_label) if true_label is not None else None
                conf = float(pred[0][pred_label_idx])
                raw_results.append({
                    'filename': fname,
                    'pred_label': pred_label,
                    'confidence': conf,
                    'true_label': true_label if true_label is not None else '',
                    'is_correct': is_correct
                })
            except Exception as e:
                raw_results.append({'filename': fname, 'error': str(e)})
        total = [r for r in raw_results if 'is_correct' in r and r['is_correct'] is not None]
        raw_acc = np.mean([r['is_correct'] for r in total]) if len(total) > 0 else None
        pd.DataFrame(raw_results).to_csv(random_test_csv_path, index=False)
        print('[INFO] raw_dataset 평가 결과 saved:', random_test_csv_path)
        print('[INFO] raw_dataset 정확도:', raw_acc)
    else:
        print('[INFO] raw_dataset 폴더에 평가 가능한 이미지가 없습니다. 평가를 건너뜁니다.')
else:
    print('[INFO] raw_dataset 폴더가 존재하지 않습니다. 평가를 건너뜁니다.')

if experiment_history and isinstance(experiment_history, list):
    try:
        pd.DataFrame(experiment_history).to_csv(experiment_report_path, index=False)
    except Exception:
        pass