```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint

# 데이터 경로 설정
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'
img_height = 224
img_width = 224
batch_size = 32

# 데이터 증강 및 제너레이터 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# 모델 구성
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
checkpoint = ModelCheckpoint(
    'best_model_colab.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv')

callbacks = [checkpoint, reduce_lr, csv_logger]

# 모델 학습
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=callbacks
)

# 테스트 데이터 예측
test_generator.reset()
pred_probs = model.predict(test_generator, verbose=1)
pred_classes = np.argmax(pred_probs, axis=1)

filenames = test_generator.filenames
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

results_df = pd.DataFrame({
    'filename': filenames,
    'true_label': [class_labels[i] for i in true_classes],
    'pred_label': [class_labels[i] for i in pred_classes],
    'correct': (true_classes == pred_classes).astype(int)
})
results_df.to_csv('random_test_results.csv', index=False)

# 학습 기록 저장 (CSVLogger가 이미 history_colab.csv를 저장함)
```