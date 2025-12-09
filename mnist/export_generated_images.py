import torch
import os
from torchvision.utils import save_image

# Load the reduced dataset (100 per class)
data = torch.load("generated_data_100.pt")      # shape: [1000, 1, 28, 28]
labels = torch.load("generated_label_100.pt")   # shape: [1000, 10]

print("Loaded:")
print(" data:", data.shape)
print(" labels:", labels.shape)

# Output root folder
root = "generated_images_100_sorted"
os.makedirs(root, exist_ok=True)

# Create subfolders 0–9
for d in range(10):
    os.makedirs(os.path.join(root, str(d)), exist_ok=True)

# Export images
for i in range(data.shape[0]):
    img = data[i]       # [1, 28, 28]
    label = torch.argmax(labels[i]).item()

    folder = os.path.join(root, str(label))
    fname = f"{i:04d}.png"

    save_image(img, os.path.join(folder, fname))

    if i % 100 == 0:
        print(f"Saved {i}/{data.shape[0]}")

print("Done! Images saved in 'generated_images_100_sorted/'")
