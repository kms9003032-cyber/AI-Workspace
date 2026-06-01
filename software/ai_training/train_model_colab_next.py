import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42

def color_jitter(img):
    if np.random.rand()<0.8:
        factor = np.random.uniform(0.7,1.3)
        img = img * factor
        img = np.clip(img,0,1)
    return img

def gaussian_noise(img):
    if np.random.rand()<0.3:
        noise = np.random.normal(0,0.03,img.shape)
        img = img+noise
        img = np.clip(img,0,1)
    return img

def random_contrast(img):
    if np.random.rand()<0.8:
        factor = np.random.uniform(0.7,1.3)
        mean = img.mean(axis=(0,1),keepdims=True)
        img = (img-mean)*factor+mean
        img = np.clip(img,0,1)
    return img

def random_cutout(img):
    if np.random.rand()<0.3:
        h,w,_ = img.shape
        cutout_size = np.random.randint(h//8,h//4)
        x = np.random.randint(0,w-cutout_size)
        y = np.random.randint(0,h-cutout_size)
        img[y:y+cutout_size,x:x+cutout_size,:]=img.mean()
    return img

def custom_preprocessing(img):
    img = color_jitter(img)
    img = gaussian_noise(img)
    img = random_contrast(img)
    img = random_cutout(img)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.35,
    shear_range=0.12,
    brightness_range=[0.65,1.35],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=custom_preprocessing
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices

train_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i:w for i,w in enumerate(class_weights)}

experiment_history = {}
load_continue = False
best_model_path = os.path.join(base_dir, 'best_model_colab.keras')
if os.path.exists(best_model_path):
    try:
        previous_model = load_model(best_model_path, compile=False)
        output_shape = previous_model.output.shape[-1]
        if output_shape==num_classes:
            base_model = previous_model.layers[1]
            load_continue = True
        else:
            experiment_history['reinit_reason'] = f'class mismatch: prev:{output_shape}, now:{num_classes}'
            load_continue=False
    except Exception as e:
        experiment_history['reinit_reason'] = str(e)
        load_continue=False

if load_continue:
    model = load_model(best_model_path)
else:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)

model.compile(optimizer=Adam(learning_rate=2e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
final_model_path = os.path.join(base_dir, 'final_model_colab.keras')
csvlogger_cb = CSVLogger(os.path.join(base_dir,'history_colab.csv'), append=False)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

callbacks=[checkpoint_cb, csvlogger_cb, reduce_lr_cb]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=2
)

model.save(final_model_path)

# RAW DATASET 평가
def allowed_img(fname):
    fname_low = fname.lower()
    return fname_low.endswith('.jpg') or fname_low.endswith('.jpeg') or fname_low.endswith('.png')

if os.path.exists(raw_test_dir) and len([x for x in os.listdir(raw_test_dir) if allowed_img(x)])>0:
    img_list = [x for x in os.listdir(raw_test_dir) if allowed_img(x)]
    results = []
    for fname in img_list:
        try:
            img_path = os.path.join(raw_test_dir, fname)
            img = load_img(img_path, target_size=IMG_SIZE)
            arr = img_to_array(img)/255.0
            arr = arr[np.newaxis,...]
            pred = model.predict(arr)
            pred_idx = np.argmax(pred)
            confidence = float(np.max(pred))
            pred_label = [k for k,v in class_indices.items() if v==pred_idx][0]
            results.append({'filename':fname, 'pred_class':pred_label, 'confidence':confidence})
        except Exception as e:
            results.append({'filename':fname, 'pred_class':'error', 'confidence':0.0})
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'random_test_results.csv'), index=False, encoding='utf-8-sig')
else:
    experiment_history['raw_dataset_eval'] = 'skipped'

if len(experiment_history)>0:
    hist_path = os.path.join(base_dir, 'experiment_metadata.csv')
    pd.DataFrame([experiment_history]).to_csv(hist_path, index=False)