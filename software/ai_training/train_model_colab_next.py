import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import glob

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
img_size = (224, 224)
batch_size = 32
epochs = 100
history_csv_path = 'history_colab.csv'
best_model_path = 'best_model_colab.keras'
final_model_path = 'final_model_colab.keras'
random_test_result_csv = 'random_test_results.csv'
experiment_history = []

def get_num_classes(directory):
    return len([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.18,
    height_shift_range=0.18,
    shear_range=0.10,
    zoom_range=[0.7, 1.28],
    brightness_range=[0.58, 1.32],
    channel_shift_range=28.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_gen.class_indices
index_to_class = dict((v,k) for k,v in class_indices.items())
num_classes = train_gen.num_classes

y_train = train_gen.classes
class_weights_raw = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight = dict(zip(np.unique(y_train), class_weights_raw))

model_created_from_scratch = False
model_loaded = False

if os.path.exists(best_model_path):
    try:
        loaded_model = load_model(best_model_path)
        if loaded_model.output_shape[-1] == num_classes:
            model = loaded_model
            model_loaded = True
            experiment_history.append('Load model: best_model_colab.keras resumed')
        else:
            model_created_from_scratch = True
            experiment_history.append('best_model_colab.keras exists but class count changed: model recreated')
    except Exception as e:
        model_created_from_scratch = True
        experiment_history.append(f'Cannot load best_model_colab.keras: model recreated, error={e}')
else:
    model_created_from_scratch = True
    experiment_history.append('best_model_colab.keras not found: model created from scratch')

if not model_loaded:
    base_model = MobileNetV2(input_shape=img_size+(3,), include_top=False, weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.42)(x)
    x = Dense(192, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.33)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

optimizer = Adam(learning_rate=2e-4)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mcp = ModelCheckpoint(
    best_model_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
csvlogger = CSVLogger(history_csv_path, append=False)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
callbacks = [mcp, csvlogger, rlr]

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save(final_model_path)

if os.path.exists(best_model_path):
    best_model = load_model(best_model_path)
else:
    best_model = model

if os.path.isdir(raw_test_dir):
    img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(raw_test_dir) if os.path.splitext(f)[1].lower() in img_extensions]
    if len(image_files) > 0:
        results = []
        for fname in image_files:
            try:
                img_path = os.path.join(raw_test_dir, fname)
                img = load_img(img_path, target_size=img_size)
                x = img_to_array(img)
                x = x / 255.0
                x = np.expand_dims(x, axis=0)
                pred = best_model.predict(x)
                top_idx = np.argmax(pred)
                top_class = index_to_class[top_idx]
                top_conf = float(pred[0][top_idx])
                result_row = {'filename': fname, 'pred_class': top_class, 'pred_conf': top_conf}
                for idx, cname in index_to_class.items():
                    result_row[f'conf_{cname}'] = float(pred[0][idx])
                results.append(result_row)
            except Exception as e:
                continue
        pd.DataFrame(results).to_csv(random_test_result_csv, index=False)
else:
    pass

with open('experiment_history.txt', 'a') as f:
    for line in experiment_history:
        f.write(f"{line}\n")