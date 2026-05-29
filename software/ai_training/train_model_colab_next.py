import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'raw_dataset')

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
SEED = 42
EPOCHS = 100

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    shear_range=0.18,
    zoom_range=[0.8, 1.2],
    width_shift_range=0.18,
    height_shift_range=0.18,
    brightness_range=[0.65, 1.35],
    channel_shift_range=15.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)

class_indices = train_gen.class_indices
classes = list(class_indices.keys())
labels = []
for klass in classes:
    labels += [klass] * len(os.listdir(os.path.join(train_dir, klass)))
label_to_index = {label: idx for idx, label in enumerate(classes)}
y = [label_to_index[label] for label in labels]
class_weights = compute_class_weight('balanced', classes=np.arange(len(classes)), y=y)
class_weights = dict(enumerate(class_weights))

base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
base_model.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cp_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai'
best_model_path = os.path.join(cp_dir, 'best_model_colab.keras')
final_model_path = os.path.join(cp_dir, 'final_model_colab.keras')

mcp = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

csv_logger = CSVLogger(os.path.join(cp_dir, 'history_colab.csv'), append=False)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.35,
    patience=4,
    min_lr=3e-6,
    verbose=1
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[mcp, csv_logger, reduce_lr]
)

model.save(final_model_path)

# 랜덤 평가 셋 예측 및 결과 저장
random_test_gen = val_aug.flow_from_directory(
    random_test_dir,
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
    seed=SEED
)
model_best = tf.keras.models.load_model(best_model_path)
preds = model_best.predict(random_test_gen, verbose=1)
y_true = random_test_gen.classes
y_pred = np.argmax(preds, axis=1)
class_labels = list(random_test_gen.class_indices.keys())
report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(cp_dir, 'random_test_results.csv'))

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(cp_dir, 'history_colab.csv'), index=False)