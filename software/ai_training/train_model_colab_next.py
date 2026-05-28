import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
random_test_dir = os.path.join(base_dir, 'random_test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 17

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=[0.85, 1.15],
    brightness_range=[0.65, 1.35],
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=25.0,
    fill_mode="nearest"
)

val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_gen = val_aug.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

num_classes = train_gen.num_classes
labels = train_gen.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=IMG_SIZE+(3,))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(2e-4))(x)
model = Model(inputs=base_model.input, outputs=out)

model.compile(
    optimizer=Adam(learning_rate=0.0007),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
csvlogger_cb = CSVLogger('history_colab.csv')
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint_cb, csvlogger_cb, reduce_lr_cb]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=2
)

model.save('final_model_colab.keras')

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_colab.csv', index=False)

random_test_gen = val_aug.flow_from_directory(
    random_test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

model.load_weights('best_model_colab.keras')

random_preds = model.predict(random_test_gen, verbose=1)
random_pred_labels = np.argmax(random_preds, axis=1)
label_map = {v: k for k, v in random_test_gen.class_indices.items()}

filenames = random_test_gen.filenames
results = []
for i, fname in enumerate(filenames):
    true_class = label_map[np.argmax(random_test_gen[i][1])]
    pred_class = label_map[random_pred_labels[i]]
    top1_prob = float(np.max(random_preds[i]))
    results.append({
        'filename': fname,
        'true_class': true_class,
        'pred_class': pred_class,
        'top1_prob': top1_prob
    })

results_df = pd.DataFrame(results)
results_df.to_csv('random_test_results.csv', index=False)