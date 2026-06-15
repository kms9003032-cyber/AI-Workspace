import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import cv2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
input_shape = (224, 224, 3)
EPOCHS = 100
BATCH_SIZE = 32
best_model_path = 'best_model_colab.keras'
final_model_path = 'final_model_colab.keras'
history_csv = 'history_colab.csv'
random_test_csv = 'random_test_results.csv'
experiment_history = []

def get_classes_and_weights(train_dir):
    class_list = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_list)
    class_counts = []
    for c in class_list:
        c_path = os.path.join(train_dir, c)
        count = len([f for f in os.listdir(c_path) if os.path.isfile(os.path.join(c_path, f))])
        class_counts.append(count)
    max_count = max(class_counts)
    class_weight = {i: float(max_count)/c if c > 0 else 1.0 for i, c in enumerate(class_counts)}
    return class_list, num_classes, class_weight

classes, num_classes, class_weight = get_classes_and_weights(train_dir)

train_augment = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.2,
    brightness_range=[0.65, 1.5],
    channel_shift_range=28.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_augment.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

resume_from_best = False
if os.path.exists(best_model_path):
    try:
        loaded_model = load_model(best_model_path)
        loaded_output_shape = loaded_model.output.shape[-1]
        if loaded_output_shape == num_classes:
            model = loaded_model
            resume_from_best = True
        else:
            experiment_history.append("output_shape_mismatch_new_model")
            model = None
    except Exception as e:
        experiment_history.append("load_model_failed_new_model")
        model = None
else:
    experiment_history.append("no_previous_best_new_model")
    model = None

if not resume_from_best:
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    experiment_history.append("build_new_mobilenetv2_model")

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=5, verbose=1, min_lr=1e-6
)
csv_logger = CSVLogger(history_csv)
checkpoint = ModelCheckpoint(
    best_model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False
)

callbacks = [checkpoint, csv_logger, lr_scheduler]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save(final_model_path)

if os.path.exists(best_model_path):
    best_model = load_model(best_model_path)
else:
    best_model = model

if os.path.isdir(raw_test_dir):
    raw_img_files = [
        f for f in os.listdir(raw_test_dir)
        if os.path.isfile(os.path.join(raw_test_dir, f)) and
           f.lower().split('.')[-1] in ['jpg', 'jpeg', 'png']
    ]
    if len(raw_img_files) > 0:
        label_map = train_gen.class_indices
        rev_label_map = {v: k for k, v in label_map.items()}
        y_pred, y_score, filenames = [], [], []
        for fname in raw_img_files:
            img_path = os.path.join(raw_test_dir, fname)
            try:
                img = load_img(img_path, target_size=input_shape[:2])
                arr = img_to_array(img).astype('float32') / 255.
                arr = np.expand_dims(arr, axis=0)
                pred = best_model.predict(arr, verbose=0)
                label_idx = int(np.argmax(pred))
                label_conf = float(np.max(pred))
                y_pred.append(label_idx)
                y_score.append(label_conf)
                filenames.append(fname)
            except Exception as e:
                filenames.append(fname)
                y_pred.append(-1)
                y_score.append(0.0)
        pred_names = [rev_label_map.get(i, 'unknown') if i != -1 else 'error' for i in y_pred]
        results_df = pd.DataFrame({
            'filename': filenames,
            'y_pred_idx': y_pred,
            'y_pred_class': pred_names,
            'confidence': y_score
        })
        results_df.to_csv(random_test_csv, index=False)
else:
    pass