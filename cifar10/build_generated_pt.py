import os
import torch
from torchvision import transforms
from PIL import Image

# Path to the folder containing the 10 class folders
BASE_PATH = "generated_data (stable_diffusion)"

# CIFAR-10 class order
CLASSES = [
    "generated_airplane",
    "generated_automobile",
    "generated_bird",
    "generated_cat",
    "generated_deer",
    "generated_dog",
    "generated_frog",
    "generated_horse",
    "generated_ship",
    "generated_truck"
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0),(1,1,1))  # match your victim training script
])

all_images = []
all_labels = []

print("[INFO] Loading generated CIFAR10 class folders...")

for class_idx, class_name in enumerate(CLASSES):
    folder_path = os.path.join(BASE_PATH, class_name)

    if not os.path.isdir(folder_path):
        print(f"[WARNING] Folder missing: {folder_path}")
        continue

    print(f"[INFO] Loading class {class_idx}: {class_name}")

    for file in os.listdir(folder_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(folder_path, file)

            img = Image.open(img_path).convert("RGB")
            img = transform(img)

            all_images.append(img)
            all_labels.append(class_idx)

print(f"[INFO] Total loaded images: {len(all_images)}")

# Convert to tensors
generated_data = torch.stack(all_images)       # (N, 3, 32, 32)
generated_label = torch.tensor(all_labels)     # (N,)

# Save
torch.save(generated_data, "generated_data.pt")
torch.save(generated_label, "generated_label.pt")

print("[DONE] Saved generated_data.pt and generated_label.pt")
