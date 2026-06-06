import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import cv2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
img_size = (160, 160)
batch_size = 32
epochs = 100
seed = 27

class_names = sorted(next(os.walk(train_dir))[1])
num_classes = len(class_names)
class_indices = {cls: i for i, cls in enumerate(class_names)}
train_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in class_names]
class_weight = compute_class_weight('balanced', classes=np.arange(num_classes), y=np.concatenate([[i]*n for i, n in enumerate(train_counts)]))
class_weight = {i:w for i,w in enumerate(class_weight)}

def strong_augment(x):
    if np.random.rand() < 0.4:
        degree = np.random.choice([3, 5])
        x = cv2.GaussianBlur(x, (degree,degree), 0)
    if np.random.rand() < 0.5:
        rate = np.random.uniform(0.6,1.0)
        mask = np.random.binomial(1, rate, x.shape)
        x = x * mask
    if np.random.rand() < 0.4:
        factor = np.random.uniform(0.8,1.2)
        x = x * factor
    if np.random.rand() < 0.5:
        y1 = np.random.randint(0, x.shape[0] - 20)
        x1 = np.random.randint(0, x.shape[1] - 20)
        x[y1:y1+20,x1:x1+20,:] = 0
    return np.clip(x,0,255).astype('uint8')

def preprocessing(img):
    img = img.astype(np.float32)/255.0
    if np.random.rand()<0.2:
        noise = np.random.normal(0, 0.03, img.shape)
        img += noise
    if np.random.rand()<0.3:
        y1 = np.random.randint(0, img.shape[0] - 10)
        x1 = np.random.randint(0, img.shape[1] - 10)
        img[y1:y1+10,x1:x1+10,:] = 0
    img = np.clip(img, 0, 1)
    img = (img*255).astype('uint8')
    return img

class CustomGen(tf.keras.utils.Sequence):
    def __init__(self, gen, unknown_classes, strong_aug_prob=0.21):
        self.gen = gen
        self.unknown_classes = unknown_classes
        self.strong_aug_prob = strong_aug_prob
        self.class_indices_rev = {v: k for k, v in gen.class_indices.items()}
    def __len__(self):
        return len(self.gen)
    def __getitem__(self, idx):
        batch_x, batch_y = self.gen[idx]
        for i in range(len(batch_x)):
            label_idx = np.argmax(batch_y[i])
            label_class = self.class_indices_rev[label_idx]
            if any(unk in label_class for unk in self.unknown_classes):
                if np.random.rand() < self.strong_aug_prob:
                    batch_x[i] = strong_augment(batch_x[i])
            else:
                batch_x[i] = preprocessing(batch_x[i])
        batch_x = batch_x.astype('float32')/255.0
        return batch_x, batch_y

train_idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.13,
    height_shift_range=0.14,
    shear_range=0.11,
    zoom_range=[0.87,1.14],
    brightness_range=[0.6,1.4],
    channel_shift_range=22.0,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True
)
val_idg = ImageDataGenerator(rescale=1./255)

train_gen = train_idg.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)
val_gen = val_idg.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

train_gen = CustomGen(train_gen, unknown_classes=['unknown','empty'])
steps_per_epoch = train_gen.__len__()
validation_steps = val_gen.n // batch_size

def build_model(num_classes):
    base = MobileNetV2(input_shape=img_size+(3,), include_top=False, weights='imagenet')
    base.trainable = True
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(base.input, out)

start_from_scratch = False
load_path = 'best_model_colab.keras'
model = None
experiment_history = []
if os.path.exists(load_path):
    try:
        tm = load_model(load_path)
        if tm.output_shape[-1]==num_classes:
            model = tm
            experiment_history.append('Continued from best_model_colab.keras')
        else:
            start_from_scratch = True
            experiment_history.append('Class count mismatch; new model')
    except Exception as e:
        start_from_scratch = True
        experiment_history.append(f'load_model error: {e}; create new model')
if model is None or start_from_scratch:
    model = build_model(num_classes)
    experiment_history.append('Start from scratch (new MobileNetV2)')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

ckpt = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv', append=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.25,
    patience=8,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)

hist = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=[ckpt, csv_logger, reduce_lr],
    verbose=1,
    initial_epoch=0
)

model.save('final_model_colab.keras')
experiment_history_path = 'experiment_history_log.txt'
with open(experiment_history_path, 'a') as f:
    for line in experiment_history:
        f.write(f"{line}\n")

history_df = pd.DataFrame(hist.history)
history_df.to_csv('history_colab.csv', mode='a', index=False)

if os.path.isdir(raw_test_dir):
    allow_ext = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG')
    img_files = []
    for f in os.listdir(raw_test_dir):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            img_files.append(os.path.join(raw_test_dir, f))
    results = []
    for fp in img_files:
        try:
            img = cv2.imread(fp)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            arr = img.astype('float32') / 255.0
            arr = np.expand_dims(arr,0)
            prob = model.predict(arr, verbose=0)[0]
            pred_idx = np.argmax(prob)
            pred_class = class_names[pred_idx]
            conf = float(prob[pred_idx])
            results.append({'file':fp,'pred_label':pred_class,'confidence':conf})
        except Exception as e:
            results.append({'file':fp,'pred_label':'error','confidence':-1})
    pd.DataFrame(results).to_csv('random_test_results.csv',index=False)
else:
    pass
