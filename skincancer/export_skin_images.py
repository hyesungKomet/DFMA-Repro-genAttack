import torch
import os
from torchvision.utils import save_image

# Load generated data
data = torch.load("generated_data.pt")         # shape ~ [40000, 3, 64, 64]
labels = torch.load("generated_label.pt")      # shape ~ [40000, 2]

print("Loaded:")
print(" data:", data.shape)
print(" labels:", labels.shape)

# Predict class from victim model outputs
pred_class = torch.argmax(labels, dim=1)       # 0 = benign, 1 = malignant

# Separate indices
benign_idx = (pred_class == 0).nonzero().squeeze()
malignant_idx = (pred_class == 1).nonzero().squeeze()

print(f"Benign samples: {len(benign_idx)}")
print(f"Malignant samples: {len(malignant_idx)}")

# Choose 500 each (random)
benign_idx = benign_idx[:500]
malignant_idx = malignant_idx[:500]

# Output folders
root = "skin_generated_500"
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, "benign"), exist_ok=True)
os.makedirs(os.path.join(root, "malignant"), exist_ok=True)

# Save benign images
for i, idx in enumerate(benign_idx):
    img = data[idx]
    save_image(img, f"{root}/benign/{i:04d}.png")

# Save malignant images
for i, idx in enumerate(malignant_idx):
    img = data[idx]
    save_image(img, f"{root}/malignant/{i:04d}.png")

print("Finished exporting 500 benign + 500 malignant images.")
