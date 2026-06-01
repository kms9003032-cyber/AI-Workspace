import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
raw_test_dir = os.path.join(base_dir, "raw_dataset")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
best_model_path = os.path.join(models_dir, "best_model_colab.keras")
final_model_path = os.path.join(models_dir, "final_model_colab.keras")
history_csv_path = os.path.join(models_dir, "history_colab.csv")
raw_test_results_csv = os.path.join(models_dir, "random_test_results.csv")
experiment_report_path = os.path.join(models_dir, "experiment_report.txt")

img_size = (224, 224)
batch_size = 32
epochs = 100
seed = 333

experiment_history = []
resume_training = False
initial_val_acc = None
best_val_acc = None
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
class_indices_dict = {v: k for k, v in enumerate(class_names)}

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=180,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=[0.85, 1.15],
    shear_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

y_train = train_generator.classes
computed_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(computed_class_weights))

initial_epoch = 0

if os.path.exists(best_model_path):
    try:
        model = load_model(best_model_path)
        resume_training = True
        if os.path.exists(history_csv_path):
            try:
                prev_history = pd.read_csv(history_csv_path)
                if 'val_accuracy' in prev_history.columns:
                    initial_val_acc = prev_history['val_accuracy'].max()
                    best_val_acc = float(initial_val_acc)
            except Exception as e:
                initial_val_acc = None
                best_val_acc = None
        trainable_count = np.sum([w.trainable for w in model.layers])
        if hasattr(model, "class_names"):
            model_class_names = getattr(model, "class_names", class_names)
            if sorted(list(model_class_names)) != sorted(class_names):
                model = None
                resume_training = False
                experiment_history.append("Class labels mismatch: Rebuilding model.")
        if (model is not None) and (model.output_shape[-1] != num_classes):
            model = None
            resume_training = False
            experiment_history.append("Output class count mismatch: Rebuilding model.")
        if model is not None:
            experiment_history.append(f"Resumed training from previous best model. Previous best val_accuracy={best_val_acc}")
    except Exception as e:
        model = None
        resume_training = False
        experiment_history.append(f"Failed to resume from previous best model. Starting new model. Error: {str(e)}")
else:
    model = None
    experiment_history.append("No best_model_colab.keras found. Training from scratch.")

if model is None:
    mobilenetv2_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    mobilenetv2_base.trainable = True
    x = GlobalAveragePooling2D()(mobilenetv2_base.output)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=mobilenetv2_base.input, outputs=output_tensor)
    model.class_names = class_names
    resume_training = False
    initial_val_acc = None
    best_val_acc = None
    experiment_history.append("Created new MobileNetV2 model from scratch.")

model.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_cb = ModelCheckpoint(
    best_model_path,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)
logger_cb = CSVLogger(history_csv_path, append=resume_training)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=8,
    verbose=1,
    min_lr=2e-6
)

callbacks = [checkpoint_cb, logger_cb, reduce_lr_cb]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=callbacks,
    class_weight=class_weights,
    initial_epoch=initial_epoch,
    verbose=1
)

model.save(final_model_path)

def predict_raw_dataset(model, class_names, test_dir, csv_save_path):
    if not os.path.exists(test_dir):
        return None
    img_list = []
    file_list = []
    for fname in os.listdir(test_dir):
        f_lower = fname.lower()
        if any([ext in f_lower for ext in ['.jpg', '.jpeg', '.png']]):
            try:
                path = os.path.join(test_dir, fname)
                img = load_img(path, target_size=img_size)
                arr = img_to_array(img)
                arr = arr / 255.0
                arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                img_list.append(arr)
                file_list.append(fname)
            except Exception as e:
                continue
    if len(img_list) == 0:
        return None
    imgs = np.stack(img_list)
    preds = model.predict(imgs, batch_size=8, verbose=1)
    top1_indices = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    resultlist = []
    for fname, idx, conf in zip(file_list, top1_indices, confidences):
        resultlist.append({
            "filename": fname,
            "predicted_label": class_names[idx],
            "confidence": conf
        })
    df = pd.DataFrame(resultlist)
    df.to_csv(csv_save_path, index=False)
    total = len(df)
    acc = None
    if "label" in df.columns:
        acc = (df["label"] == df["predicted_label"]).mean()
    counter = df["predicted_label"].value_counts().to_dict()
    experiment_history.append(f"raw_dataset total: {total}, class_count: {counter}, accuracy: {acc}")
    return df

raw_eval_result = predict_raw_dataset(model, class_names, raw_test_dir, raw_test_results_csv)

try:
    if os.path.exists(history_csv_path):
        hist_df = pd.read_csv(history_csv_path)
        if 'val_accuracy' in hist_df.columns:
            last_val_acc = hist_df['val_accuracy'].dropna().values[-1]
            experiment_history.append(f"Final val_accuracy: {last_val_acc:.4f}")
        if best_val_acc is not None:
            experiment_history.append(f"Previous best val_accuracy: {best_val_acc:.4f}")
except Exception as e:
    pass

with open(experiment_report_path, "w", encoding="utf-8") as f:
    for line in experiment_history:
        f.write(line+"\n")

if raw_eval_result is not None:
    print(f"raw_dataset 평가 결과 저장: {raw_test_results_csv}")
else:
    print("raw_dataset 없음 → 평가 건너뜀.")

print("학습 이력, experiment_report.txt 저장 완료.")
print("Best model path:", best_model_path)
print("Final model path:", final_model_path)
print("history_colab.csv path:", history_csv_path)
if os.path.exists(raw_test_results_csv):
    print("random_test_results.csv path:", raw_test_results_csv)