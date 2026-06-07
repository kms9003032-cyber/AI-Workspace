import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import Counter

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 100
SEED = 42
HISTORY_CSV = 'history_colab.csv'
RAW_RESULTS_CSV = 'random_test_results.csv'
CHECKPOINT_FILE = 'best_model_colab.keras'
FINAL_MODEL_FILE = 'final_model_colab.keras'
LOG_FILE = 'history_colab.csv'
experiment_history = []

def calc_class_weights(generator):
    cls = generator.classes
    cnt = Counter(cls)
    max_cnt = float(max(cnt.values()))
    return {k: max_cnt / v for k, v in cnt.items()}

def strong_augment(img):
    img = img.astype(np.float32)
    img += np.random.normal(0, 10, img.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.7,1.3)
        img = img * factor
    if np.random.rand() < 0.5:
        img = img + np.random.uniform(-20,20)
    img = np.clip(img, 0, 255)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.18,
    height_shift_range=0.18,
    shear_range=0.13,
    zoom_range=0.18,
    brightness_range=[0.72, 1.32],
    channel_shift_range=32.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=strong_augment
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.92,1.08]
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices
inv_class_indices = {v:k for k,v in class_indices.items()}
class_weights = calc_class_weights(train_gen)

initial_model_exists = os.path.exists(CHECKPOINT_FILE)
load_succeeded = False
model = None

if initial_model_exists:
    try:
        temp_model = load_model(CHECKPOINT_FILE)
        output_shape = temp_model.output.shape[-1]
        if output_shape == num_classes:
            model = temp_model
            load_succeeded = True
            experiment_history.append({'event':'resume','note':'Resume from existing best_model_colab.keras'})
    except Exception as e:
        experiment_history.append({'event':'fail_resume','note':str(e)})

if not load_succeeded:
    base_model = MobileNetV2(input_shape=IMG_SIZE+(3,),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, output)
    experiment_history.append({'event':'new_model','note':'init new MobileNetV2'})
    
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint(CHECKPOINT_FILE, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
csv_logger = CSVLogger(LOG_FILE, append=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.36, patience=5, min_lr=5e-6, verbose=1)

try:
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, csv_logger, reduce_lr_cb]
    )
    pd.DataFrame(history.history).to_csv(HISTORY_CSV, index=False)
except Exception as e:
    experiment_history.append({'event':'fit_exception','note':str(e)})

try:
    model = load_model(CHECKPOINT_FILE)
except:
    pass

try:
    model.save(FINAL_MODEL_FILE)
except:
    pass

def is_image_file(fn):
    ext = fn.lower().split('.')[-1]
    return ext in ['jpg','jpeg','png']

if os.path.isdir(raw_test_dir):
    raw_fnames = [f for f in os.listdir(raw_test_dir) if is_image_file(f)]
    if len(raw_fnames)>0:
        results = []
        for fname in raw_fnames:
            fpath = os.path.join(raw_test_dir, fname)
            try:
                img = load_img(fpath, target_size=IMG_SIZE)
                arr = img_to_array(img)/255.0
                arr = np.expand_dims(arr,0)
                pred = model.predict(arr,verbose=0)[0]
                pred_idx = np.argmax(pred)
                pred_class = inv_class_indices[pred_idx]
                confidence = float(pred[pred_idx])
                results.append({'filename':fname,'pred_class':pred_class,
                                'confidence':confidence})
            except Exception as e:
                results.append({'filename':fname,'pred_class':'error','confidence':-1,'error':str(e)})
        df_r = pd.DataFrame(results)
        df_r.to_csv(RAW_RESULTS_CSV,index=False)
else:
    experiment_history.append({'event':'skip_raw_dataset_evaluation','note':'No raw_dataset directory'})

if experiment_history:
    pd.DataFrame(experiment_history).to_csv("experiment_history.csv",index=False)