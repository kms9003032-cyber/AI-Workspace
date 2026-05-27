```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import pandas as pd
import numpy as np

# 경로 설정
train_dir = 'data/train'
valid_dir = 'data/valid'
test_dir = 'data/test'

# 파라미터
img_height, img_width = 224, 224
batch_size = 32
epochs = 100

# 데이터 생성기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
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

num_classes = train_generator.num_classes

# 모델 구축
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=1e-6, 
    verbose=1
)
csv_logger = CSVLogger('history_colab.csv')
checkpoint = ModelCheckpoint(
    'best_model_colab.keras', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

callbacks = [reduce_lr, csv_logger, checkpoint]

# 학습
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=epochs,
    callbacks=callbacks
)

# 테스트
model.load_weights('best_model_colab.keras')
test_generator.reset()
preds = model.predict(
    test_generator, 
    steps=len(test_generator), 
    verbose=1
)
predicted_class_indices = np.argmax(preds, axis=1)
true_class_indices = test_generator.classes
filenames = test_generator.filenames
class_labels = list(test_generator.class_indices.keys())

results_df = pd.DataFrame({
    'filename': filenames,
    'actual': [class_labels[i] for i in true_class_indices],
    'predicted': [class_labels[i] for i in predicted_class_indices]
})
results_df.to_csv('random_test_results.csv', index=False)
```