import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ==============================
# 기본 경로 설정
# ==============================

base_dir = r"./dataset"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

model_dir = r"./models"
result_dir = r"./training_results"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ==============================
# 데이터 증강
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==============================
# 데이터 로딩
# ==============================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ==============================
# 모델 생성
# ==============================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# ==============================
# 컴파일
# ==============================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 콜백 설정
# ==============================

csv_logger = CSVLogger(
    os.path.join(result_dir, "training_log.csv"),
    append=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==============================
# 학습 시작
# ==============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[
        csv_logger,
        reduce_lr,
        checkpoint
    ]
)

# ==============================
# 최종 모델 저장
# ==============================

model.save(
    os.path.join(model_dir, "final_model.keras")
)

# ==============================
# 결과 저장
# ==============================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    os.path.join(result_dir, "history.csv"),
    index=False
)

print("Training Finished")
