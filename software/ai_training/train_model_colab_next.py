import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing import image

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
model_dir = os.path.join(base_dir, 'models')

best_model_path = os.path.join(model_dir, 'best_model_colab.keras')
final_model_path = os.path.join(model_dir, 'final_model_colab.keras')
history_csv_path = os.path.join(model_dir, 'history_colab.csv')
raw_test_csv_path = os.path.join(model_dir, 'random_test_results.csv')
experiment_report_path = os.path.join(model_dir, 'experiment_report.txt')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42

os.makedirs(model_dir, exist_ok=True)

def list_dir_classes(folder):
    classes = []
    if not os.path.isdir(folder):
        return classes
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isdir(p):
            classes.append(name)
    return sorted(classes)

train_classes = list_dir_classes(train_dir)
val_classes = list_dir_classes(val_dir)
num_classes = len(train_classes)
experiment_log = []
def log_report(msg):
    experiment_log.append(msg)
    with open(experiment_report_path, 'a') as f:
        f.write(msg + '\n')

# 이미지 증강 설정 (실환경 기준, 색상 변화 無, 로테이션 유지: rotation_range=180)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.05,
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

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

# 클래스 가중치 계산
y_train = []
for label, idx in train_gen.class_indices.items():
    class_dir = os.path.join(train_dir, label)
    y_train += [idx] * len(os.listdir(class_dir))
if len(y_train) > 0:
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train
    )
    class_weight_dict = {i: cw[i] for i in range(num_classes)}
else:
    class_weight_dict = None

# 이어학습 또는 새모델 판별
model = None
start_acc = None
if os.path.exists(best_model_path):
    try:
        model = load_model(best_model_path)
        if model.output_shape[-1] == num_classes:
            log_report('[INFO] best_model_colab.keras exists. Load for fine-tuning.')
            model.compile(
                optimizer=Adam(learning_rate=1.5e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            _, start_acc = model.evaluate(val_gen, verbose=0)
            log_report(f'[INFO] Previous best validation accuracy: {start_acc:0.4f}')
        else:
            log_report('[WARNING] Output class mismatch. Creating new model.')
            model = None
    except Exception as e:
        log_report(f'[ERROR] Model load failed: {e}. Creating new model.')
        model = None

if model is None:
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        alpha=1.0,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.25)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    log_report(f'[INFO] New MobileNetV2 initialized. Classes: {num_classes}')

# 콜백 설정
checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
csvlogger_cb = CSVLogger(history_csv_path, append=True)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
callbacks = [checkpoint_cb, csvlogger_cb, reduce_lr_cb]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

model.save(final_model_path)

# history_colab.csv 갱신 (이미 append log 됨) - 별도 저장 필요시:
if hasattr(history, 'history'):
    pd.DataFrame(history.history).to_csv(history_csv_path, index=False)

# raw_dataset 평가
def evaluate_raw_dataset(model, class_indices, result_csv_path):
    if not os.path.isdir(raw_test_dir):
        log_report('[WARNING] raw_dataset folder not found. Skipped evaluation.')
        return -1
    idx_to_class = {v:k for k,v in class_indices.items()}
    files = []
    preds = []
    true_labels = []
    scores = []
    for fname in os.listdir(raw_test_dir):
        fpath = os.path.join(raw_test_dir, fname)
        if not os.path.isfile(fpath):
            continue
        fname_lower = fname.lower()
        if not ('.jpg' in fname_lower or '.jpeg' in fname_lower or '.png' in fname_lower):
            continue
        try:
            img = image.load_img(fpath, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            prob = model.predict(x, verbose=0)[0]
            pred_idx = np.argmax(prob)
            pred_label = idx_to_class[pred_idx]
            conf_score = float(prob[pred_idx])
            label = None
            # 파일명으로부터 실제 라벨 추출 (예시: label in filename)
            for cname in class_indices.keys():
                if cname.lower() in fname_lower:
                    label = cname
                    break
            is_correct = (label == pred_label) if label is not None else None
            files.append(fname)
            preds.append(pred_label)
            true_labels.append(label if label is not None else '')
            scores.append(conf_score)
        except Exception as e:
            log_report(f'[ERROR] {fname}: {e}')
            continue
    df = pd.DataFrame({
        'filename': files,
        'true_label': true_labels,
        'pred_label': preds,
        'confidence': scores
    })
    if 'true_label' in df.columns and 'pred_label' in df.columns:
        df['is_correct'] = df['true_label'] == df['pred_label']
        acc = df['is_correct'].mean()
    else:
        acc = None
    df.to_csv(result_csv_path, index=False)
    log_report(f'[RESULT] raw_dataset accuracy={acc if acc is not None else "N/A"} ({len(df)} images)')
    return acc

raw_acc = evaluate_raw_dataset(model, train_gen.class_indices, raw_test_csv_path)

if os.path.exists(best_model_path):
    try:
        best_model = load_model(best_model_path)
        _, best_val_acc = best_model.evaluate(val_gen, verbose=0)
        log_report(f'[INFO] Finished. Best val_accuracy: {best_val_acc:0.4f}')
    except:
        log_report('[INFO] Finished. (best_model skipped re-eval)')
else:
    log_report('[INFO] Finished. No previous best model.')

# 성능 급락/실패 실험 기록
if 'history' in locals() and hasattr(history, 'history'):
    val_accs = history.history.get('val_accuracy', [])
    if len(val_accs) >= 10 and max(val_accs[:10]) < 0.30:
        log_report('[FAIL] Validation accuracy stays below 0.30 at 10th epoch. Check for errors.')
if start_acc is not None and len(val_accs) > 0:
    if max(val_accs) + 0.10 < start_acc:  # 기준선보다 0.10 이상 하락
        log_report('[FAIL] Significant drop in accuracy compared to previous best.')

log_report('[INFO] Training and evaluation completed.')