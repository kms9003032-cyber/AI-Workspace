import os
import random
import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==============================
# 기본 경로 설정
# ==============================

model_path = r"./models/best_model.keras"

dataset_dir = r"./dataset"
raw_dir = os.path.join(dataset_dir, "raw_dataset")

result_dir = r"./training_results/random_test"
os.makedirs(result_dir, exist_ok=True)

summary_path = r"./training_results/random_test_results.csv"

# ==============================
# 모델 로드
# ==============================

model = load_model(model_path)

# train 폴더 기준 클래스 이름 가져오기
train_dir = os.path.join(dataset_dir, "train")
class_names = sorted(os.listdir(train_dir))

# ==============================
# 랜덤 이미지 선택
# ==============================

all_images = []

for root, dirs, files in os.walk(raw_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, file))

sample_count = min(50, len(all_images))
sample_images = random.sample(all_images, sample_count)

records = []

# ==============================
# 예측 실행
# ==============================

for img_path in sample_images:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0]

    pred_index = int(np.argmax(pred))
    pred_class = class_names[pred_index]
    confidence = float(pred[pred_index])

    save_dir = os.path.join(result_dir, pred_class)
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))

    records.append({
        "image_path": img_path,
        "predicted_class": pred_class,
        "confidence": confidence
    })

# ==============================
# 결과 저장
# ==============================

df = pd.DataFrame(records)
df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("Random image evaluation finished")
print(f"Saved results to: {summary_path}")
