import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, accuracy_score
import cv2

base_dir = '/content/drive/MyDrive/4조 응용 기초 설계/chess_piece_ai/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
raw_dir = os.path.join(base_dir, 'raw_dataset')

def add_random_shadow(img):
    h, w = img.shape[0], img.shape[1]
    top_x, bot_x = np.random.randint(w), np.random.randint(w)
    shadow_mask = np.zeros_like(img, dtype=np.uint8)
    polygon = np.array([[top_x,0],[bot_x,h],[w, h],[w,0]], np.int32)
    shadow_mask = cv2.fillPoly(shadow_mask, [polygon], (0,0,0))
    alpha = np.random.uniform(0.3, 0.75)
    img = cv2.addWeighted(img, 1, shadow_mask, alpha, 0)
    return img

def aug_fn(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = img.numpy() if isinstance(img, tf.Tensor) else img
    if np.random.rand()<0.18: img = cv2.GaussianBlur(img,(3,3),0)
    if np.random.rand()<0.17: img = np.clip(img+np.random.normal(0,0.07,img.shape),-1,1)
    if np.random.rand()<0.21:
        contrast = 1+np.random.uniform(-0.18,0.18)
        img = np.clip(img*contrast,-1,1)
    if np.random.rand()<0.15:
        brightness = np.random.uniform(-0.12,0.12)
        img = np.clip(img+brightness,-1,1)
    if np.random.rand()<0.14:
        img255 = ((img+1)*127.5).astype(np.uint8)
        img255 = add_random_shadow(img255)
        img = img255.astype(np.float32)/127.5-1
    return img

train_gen = ImageDataGenerator(
    preprocessing_function=aug_fn,
    rotation_range=180,
    width_shift_range=0.12,
    height_shift_range=0.12,
    brightness_range=(0.7,1.35),
    shear_range=0.13,
    zoom_range=0.17,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

batch_size = 32
target_size = (224,224)

train_flow = train_gen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_flow = val_gen.flow_from_directory(
    val_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
class_names = list(train_flow.class_indices.keys())
num_classes = len(class_names)

weights_df = pd.DataFrame({'cls':[],'n':[]})
for c in class_names:
    n = len(os.listdir(os.path.join(train_dir,c)))
    weights_df = pd.concat([weights_df,pd.DataFrame({'cls':[c],'n':[n]})])
class_weights_sum = {i:1/max(1,weights_df.iloc[i]['n']) for i in range(num_classes)}
class_weights = {i:class_weights_sum[i] if ('unknown' in class_names[i] or 'empty' in class_names[i]) else 1.0 for i in range(num_classes)}

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(224, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.33)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model_colab.keras',monitor='val_accuracy',save_best_only=True,mode='max',verbose=1)
csvlogger = CSVLogger('history_colab.csv')
reducelr = ReduceLROnPlateau(monitor='val_loss',factor=0.3,patience=6,min_lr=3e-6,verbose=1)
earlystop = EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True, verbose=1)

history = model.fit(
    train_flow,
    epochs=100,
    validation_data=val_flow,
    callbacks=[checkpoint,csvlogger,reducelr,earlystop],
    class_weight=class_weights,
    verbose=2
)

model.save('final_model_colab.keras')

pd.DataFrame(history.history).to_csv('history_colab.csv', index=False)

def raw_eval(model, raw_dir, class_indices, output_csv='random_test_results.csv'):
    if not os.path.exists(raw_dir):
        print('raw_dataset not found. Evaluation skipped.')
        return
    test_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    ).flow_from_directory(
        raw_dir,
        target_size=target_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    labels = test_gen.classes
    n = test_gen.samples
    y_pred = model.predict(test_gen, steps=n, verbose=1)
    pred_idx = np.argmax(y_pred,axis=1)
    inv = {v:k for k,v in class_indices.items()}
    report = classification_report(labels, pred_idx, target_names=[inv[i] for i in range(num_classes)], output_dict=True)
    acc = accuracy_score(labels, pred_idx)
    unknown_metrics = {}
    for i, c in inv.items():
        if 'unknown' in c or 'empty' in c:
            unknown_metrics[c+'_recall'] = report[c]['recall']
            unknown_metrics[c+'_f1'] = report[c]['f1-score']
    result = {'accuracy':acc, **unknown_metrics}
    pd.DataFrame([result]).to_csv(output_csv,index=False)
    with open('random_test_classification_report.txt','w') as f:
        f.write(classification_report(labels, pred_idx, target_names=[inv[i] for i in range(num_classes)]))

model = tf.keras.models.load_model('best_model_colab.keras')
raw_eval(model, raw_dir, train_flow.class_indices,'random_test_results.csv')