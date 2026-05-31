import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

import cv2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_dataset_dir = os.path.join(base_dir, 'raw_dataset')

img_height, img_width = 224, 224
batch_size = 32
epochs = 100
seed = 44

classes = sorted(next(os.walk(train_dir))[1])
num_classes = len(classes)

train_labels = []
for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    train_labels += [class_name] * len(os.listdir(class_path))
class_weights = compute_class_weight('balanced', classes=np.array(classes), y=np.array(train_labels))
class_weights_dict = dict(enumerate(class_weights))

def gaussian_noise(img):
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.03, img.shape)
        img = np.clip(img + noise, 0., 1.)
    return img

def random_blur(img):
    if np.random.rand() < 0.2:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img

def preprocess_img(img):
    img = gaussian_noise(img)
    img = random_blur(img)
    return img

def preprocessing_function(x):
    x = preprocess_img(x)
    return x

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.18,
    height_shift_range=0.18,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocessing_function,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

base_model = MobileNetV2(
    include_top=False,
    input_tensor=Input(shape=(img_height, img_width, 3)),
    weights='imagenet'
)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    filepath='best_model_colab.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
final_checkpoint = ModelCheckpoint(
    filepath='final_model_colab.keras',
    save_best_only=False,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=4,
    min_lr=1e-6,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=14,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, final_checkpoint, csv_logger, reduce_lr, early_stop]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

model.save('final_model_colab.keras')

try:
    model.load_weights('best_model_colab.keras')
except Exception:
    pass

if os.path.isdir(raw_dataset_dir):
    raw_gen = val_datagen.flow_from_directory(
        raw_dataset_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    y_true = raw_gen.classes
    y_pred_probs = model.predict(raw_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_indices = raw_gen.class_indices
    inv_class_indices = dict((v, k) for k, v in class_indices.items())

    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    unknown_idx = class_indices.get('unknown_or_empty')
    unknown_result = None
    if unknown_idx is not None:
        unknown_true = np.sum(np.array(y_true) == unknown_idx)
        unknown_pred = np.sum(np.array(y_pred) == unknown_idx)
        unknown_acc = np.mean((np.array(y_true) == unknown_idx) == (np.array(y_pred) == unknown_idx))
        unknown_result = {'unknown_true': unknown_true, 'unknown_pred': unknown_pred, 'unknown_acc': unknown_acc}

    df_rows = []
    for c in classes:
        stats = report_dict[c]
        row = {
            'class': c,
            'precision': stats['precision'],
            'recall': stats['recall'],
            'f1-score': stats['f1-score'],
            'support': stats['support']
        }
        df_rows.append(row)
    macro_micro = {
        'class': 'ALL',
        'precision': precision,
        'recall': recall,
        'f1-score': macro_f1,
        'support': np.sum([row['support'] for row in df_rows])
    }
    df_rows.append(macro_micro)
    if unknown_result is not None:
        for row in df_rows:
            row.update(unknown_result)
    df_result = pd.DataFrame(df_rows)
    df_result.to_csv('random_test_results.csv', index=False)
    np.savetxt("random_confusion_matrix.csv", cm, delimiter=",", fmt='%d')

else:
    print('/raw_dataset 폴더가 없어 랜덤 평가를 건너뜁니다.')