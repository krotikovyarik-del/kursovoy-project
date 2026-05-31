import zipfile
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns

# =========================================================
# 1. РАСПАКОВКА DATASET
# =========================================================

ZIP_PATH = "/content/im.v2i.yolov8.zip"   # <-- если имя другое, поменяй

EXTRACT_PATH = "/content/dataset"

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("Dataset extracted!")

# =========================================================
# 2. ПУТИ
# =========================================================

dataset_path = Path(EXTRACT_PATH)

splits = ["train", "valid", "test"]

records = []

# =========================================================
# 3. ЧТЕНИЕ YOLO LABELS
# =========================================================

for split in splits:

    image_dir = dataset_path / split / "images"
    label_dir = dataset_path / split / "labels"

    image_files = list(image_dir.glob("*.*"))

    print(f"{split}: {len(image_files)} images")

    for image_path in image_files:

        img = cv2.imread(str(image_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        with open(label_path, "r") as f:

            lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, bw, bh = parts

                records.append({
                    "split": split,
                    "image": image_path.name,
                    "image_width": w,
                    "image_height": h,
                    "class_id": int(class_id),
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "bbox_width": float(bw),
                    "bbox_height": float(bh),
                    "bbox_area": float(bw) * float(bh)
                })

df = pd.DataFrame(records)

# =========================================================
# 4. ОБЩАЯ ИНФОРМАЦИЯ
# =========================================================

print("\nРазмер датафрейма:")
print(df.shape)

print("\nПервые строки:")
display(df.head())

print("\nКоличество классов:")
print(df["class_id"].nunique())

print("\nКоличество объектов:")
print(len(df))

# =========================================================
# 5. ГРАФИК 1 — БАЛАНС КЛАССОВ
# =========================================================

plt.figure(figsize=(14,6))

df["class_id"].value_counts().sort_index().plot(kind="bar")

plt.title("Распределение количества объектов по классам")
plt.xlabel("Класс")
plt.ylabel("Количество")

plt.tight_layout()

plt.savefig("class_balance.png", dpi=300)

plt.show()

# =========================================================
# 6. ГРАФИК 2 — РАЗМЕРЫ ИЗОБРАЖЕНИЙ
# =========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["image_width"],
    df["image_height"],
    alpha=0.2
)

plt.title("Распределение размеров изображений")
plt.xlabel("Ширина")
plt.ylabel("Высота")

plt.tight_layout()

plt.savefig("image_sizes.png", dpi=300)

plt.show()

# =========================================================
# 7. ГРАФИК 3 — РАЗМЕРЫ BBOX
# =========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["bbox_width"],
    df["bbox_height"],
    alpha=0.2
)

plt.title("Распределение размеров bounding box")
plt.xlabel("Ширина bbox")
plt.ylabel("Высота bbox")

plt.tight_layout()

plt.savefig("bbox_sizes.png", dpi=300)

plt.show()

# =========================================================
# 8. ГРАФИК 4 — ПЛОЩАДЬ BBOX
# =========================================================

plt.figure(figsize=(8,5))

plt.hist(df["bbox_area"], bins=50)

plt.title("Распределение площади bounding box")
plt.xlabel("Площадь")
plt.ylabel("Количество объектов")

plt.tight_layout()

plt.savefig("bbox_area.png", dpi=300)

plt.show()

# =========================================================
# 9. ГРАФИК 5 — ЦЕНТРЫ ОБЪЕКТОВ
# =========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["x_center"],
    df["y_center"],
    alpha=0.2
)

plt.title("Распределение центров объектов")
plt.xlabel("X center")
plt.ylabel("Y center")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("bbox_centers.png", dpi=300)

plt.show()

# =========================================================
# 10. ПРИМЕРЫ ИЗОБРАЖЕНИЙ С BBOX
# =========================================================

sample_images = df["image"].unique()[:5]

plt.figure(figsize=(15,5))

for i, image_name in enumerate(sample_images):

    image_path = None

    for split in splits:

        p = dataset_path / split / "images" / image_name

        if p.exists():
            image_path = p
            break

    if image_path is None:
        continue

    img = cv2.imread(str(image_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w = img.shape[:2]

    sample_df = df[df["image"] == image_name]

    for _, row in sample_df.iterrows():

        x = row["x_center"] * img_w
        y = row["y_center"] * img_h

        bw = row["bbox_width"] * img_w
        bh = row["bbox_height"] * img_h

        x1 = int(x - bw / 2)
        y1 = int(y - bh / 2)

        x2 = int(x + bw / 2)
        y2 = int(y + bh / 2)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255,0,0),
            2
        )

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    plt.axis("off")

plt.tight_layout()

plt.savefig("bbox_examples.png", dpi=300)

plt.show()

print("\nГотово!")
print("Созданы графики:")
print("class_balance.png")
print("image_sizes.png")
print("bbox_sizes.png")
print("bbox_area.png")
print("bbox_centers.png")
print("bbox_examples.png")
