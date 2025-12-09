import pandas as pd
import os
import shutil
import random

# Load metadata
meta = pd.read_csv("HAM10000_metadata.csv")

# Paths to the raw image folders
img_dir1 = "HAM10000_images_part_1"
img_dir2 = "HAM10000_images_part_2"

output_root = "melanoma_cancer_dataset"

# Create required folders
for split in ["train", "test"]:
    for cls in ["benign", "malignant"]:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

# HAM10000 label categories
benign_labels = {"bkl", "nv", "vasc", "df"}
malignant_labels = {"mel", "akiec", "bcc"}

# Shuffle metadata before splitting
meta = meta.sample(frac=1, random_state=42)

# 80/20 train-test split
train_size = int(0.8 * len(meta))
train_meta = meta[:train_size]
test_meta = meta[train_size:]

def copy_images(df, split):
    for idx, row in df.iterrows():
        img_name = row["image_id"] + ".jpg"

        # Try to locate image in both folders
        img_path = os.path.join(img_dir1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir2, img_name)

        if not os.path.exists(img_path):
            print(f"Missing image: {img_name}")
            continue

        dx = row["dx"]

        if dx in benign_labels:
            cls = "benign"
        elif dx in malignant_labels:
            cls = "malignant"
        else:
            continue  # ignore unknown classes

        out_path = os.path.join(output_root, split, cls, img_name)
        shutil.copyfile(img_path, out_path)

    print(f"[DONE] Copied {len(df)} images into {split}/")

copy_images(train_meta, "train")
copy_images(test_meta, "test")

print("✨ Dataset preparation complete!")
