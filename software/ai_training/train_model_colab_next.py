import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import datetime

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_test_dir = os.path.join(base_dir, 'raw_dataset')
best_model_path = os.path.join(base_dir, 'best_model_colab.keras')
final_model_path = os.path.join(base_dir, 'final_model_colab.keras')
csv_logger_path = os.path.join(base_dir, 'history_colab.csv')
raw_eval_csv = os.path.join(base_dir, 'random_test_results.csv')
experiment_report_path = os.path.join(base_dir, 'experiment_history.txt')

batch_size = 32
img_size = (224, 224)
epochs = 100
initial_lr = 1e-3
model_load_success = False
prev_best_val_acc = None
prev_raw_acc = None
cur_init_val_acc = None

def write_report(lines):
    with open(experiment_report_path, 'a') as f:
        for line in lines:
            f.write(str(line)+'\n')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.08,
    zoom_range=0.10,
    brightness_range=(0.7,1.3),
    channel_shift_range=12.,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_gen.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
num_classes = len(class_indices)

def get_val_acc(model, val_gen):
    res = model.evaluate(val_gen, verbose=0)
    return res[1] if len(res)>1 else None

start_time = datetime.datetime.now()
report_lines = [f'[Experiment at {start_time}]']

if os.path.exists(best_model_path):
    try:
        prev_model = load_model(best_model_path)
        prev_best_val_acc = get_val_acc(prev_model, val_gen)
        prev_raw_acc = None
        if os.path.exists(raw_test_dir):
            prev_raw_acc = None
            eval_imgs = []
            eval_lbls = []
            for fname in os.listdir(raw_test_dir):
                fn_l = fname.lower()
                if fn_l.endswith(('.png','.jpg','.jpeg')):
                    path = os.path.join(raw_test_dir, fname)
                    try:
                        img = load_img(path, target_size=img_size)
                        arr = img_to_array(img)/255.
                        eval_imgs.append(arr)
                        eval_lbls.append(fname)
                    except: pass
            if len(eval_imgs)>0:
                arr = np.stack(eval_imgs)
                probs = prev_model.predict(arr)
                preds = np.argmax(probs, axis=1)
                csv_rows = []
                for i in range(len(eval_imgs)):
                    result = {'filename':eval_lbls[i],'pred':inv_class_indices[preds[i]]}
                    for k in range(num_classes):
                        result[f'prob_{inv_class_indices[k]}']=probs[i][k]
                    csv_rows.append(result)
                prev_raw_acc = None
                df = pd.DataFrame(csv_rows)
                true_count = 0
                total = 0
                for fname in eval_lbls:
                    for cname in class_indices:
                        if cname in fname:
                            if inv_class_indices[preds[eval_lbls.index(fname)]]==cname:
                                true_count+=1
                            break
                    total+=1
                if total>0:
                    prev_raw_acc = true_count/total
        report_lines.append(f'[INFO] 이어학습 시작 - best_model_colab.keras 발견')
        report_lines.append(f'[INFO] 이전 best val_accuracy: {prev_best_val_acc}')
        if prev_raw_acc:
            report_lines.append(f'[INFO] 이전 raw_dataset acc: {prev_raw_acc}')
        base_model = prev_model.layers[1]
        model = prev_model
        model_load_success = True
    except Exception as e:
        report_lines.append(f'[WARN] best_model_colab.keras 불러오기 실패: {e}')
        model_load_success = False

if not model_load_success:
    base_model = MobileNetV2(include_top=False, input_shape=(img_size[0],img_size[1],3), weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(160, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)
    model.compile(optimizer=Adam(initial_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    report_lines.append('[INFO] best_model_colab.keras로 로드 불가, 새 모델 초기화 및 이어학습 불가 사유 기록')
    model_load_success = False

checkpoint = ModelCheckpoint(
    best_model_path, monitor='val_accuracy', verbose=1,
    save_best_only=True, save_weights_only=False, mode='max'
)
csv_logger = CSVLogger(csv_logger_path)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.4, patience=6, min_lr=1e-6, verbose=2
)

if model_load_success:
    cur_init_val_acc = get_val_acc(model, val_gen)
    report_lines.append(f'[INFO] best_model_colab.keras 이어학습 전 initial val_accuracy: {cur_init_val_acc}')
else:
    report_lines.append('[INFO] 새로 학습을 시작: 이어학습 불가')

write_report(report_lines)

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[checkpoint,csv_logger,reduce_lr]
)

try:
    model.save(final_model_path)
except: pass

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(csv_logger_path, index=False)

def evaluate_on_raw(model, report_path, class_indices, inv_class_indices, save_path):
    if not os.path.exists(report_path):
        return None
    eval_imgs=[]
    eval_fnames=[]
    img_exts = ('.png','.jpg','.jpeg')
    files = [f for f in os.listdir(report_path) if f.lower().endswith(img_exts)]
    if len(files)==0:
        return None
    for fname in files:
        try:
            img = load_img(os.path.join(report_path, fname), target_size=img_size)
            arr = img_to_array(img)/255.
            eval_imgs.append(arr)
            eval_fnames.append(fname)
        except: pass
    if len(eval_imgs)==0:
        return None
    arr = np.stack(eval_imgs)
    probs = model.predict(arr, verbose=0)
    preds = np.argmax(probs, axis=1)
    results = []
    unknown_idx = None
    for k,v in class_indices.items():
        if 'unknown' in k or 'empty' in k:
            unknown_idx=v
            break
    for i in range(len(eval_imgs)):
        fn = eval_fnames[i]
        true_cls = None
        for cname in class_indices:
            if cname in fn:
                true_cls = cname
                break
        result = {
            'filename': fn,
            'predict_class': inv_class_indices[preds[i]],
            'confidence': float(np.max(probs[i])),
        }
        if true_cls:
            result['true_class']=true_cls
            result['is_correct']=int(result['predict_class']==true_cls)
        else:
            result['true_class']=None
            result['is_correct']=None
        for k in range(len(class_indices)):
            result[f'prob_{inv_class_indices[k]}']=probs[i][k]
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(save_path,index=False)
    # summary
    if 'is_correct' in df.columns and df['is_correct'].notnull().any():
        acc = df['is_correct'].dropna().astype(int).mean()
    else:
        acc = None
    unknown_pred = (df['predict_class']==inv_class_indices[unknown_idx]).sum() if unknown_idx is not None else None
    unknown_true = (df['true_class']==inv_class_indices[unknown_idx]).sum() if unknown_idx is not None and 'true_class' in df.columns else None
    return {
        'accuracy': acc,
        'unknown_pred': unknown_pred,
        'unknown_true': unknown_true,
        'total': len(df)
    }

raw_eval_result = None
if os.path.exists(raw_test_dir):
    try:
        raw_eval_result = evaluate_on_raw(model, raw_test_dir, class_indices, inv_class_indices, raw_eval_csv)
        write_report([f'[EVAL] raw_dataset result: {raw_eval_result}'])
    except Exception as e:
        write_report([f'[EVAL] raw_dataset 평가 오류: {e}'])
else:
    write_report(['[EVAL] raw_dataset 폴더 없음: 평가 생략'])

write_report(['[END]\n'])