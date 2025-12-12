import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils_cifar import get_vggmodel

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES = 3000   # keep PCA fast
BATCH_SIZE = 64

# -------------------------
# Load victim model
# -------------------------
print("[INFO] Loading victim model...")
victim = torch.load("victim_model.pt", map_location=DEVICE)
victim.to(DEVICE)
victim.eval()

# -------------------------
# Feature extractor
# -------------------------
def extract_features(model, dataloader):
    feats = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)

            # VGG feature extraction
            f = model.features(x)
            f = f.view(f.size(0), -1)

            feats.append(f.cpu())
            labels.append(y)

            if len(torch.cat(feats)) >= MAX_SAMPLES:
                break

    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

# -------------------------
# Load REAL CIFAR-10 data
# -------------------------
print("[INFO] Loading real CIFAR-10 data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1))
])

real_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

real_loader = DataLoader(
    real_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

real_feat, real_labels = extract_features(victim, real_loader)

# -------------------------
# Load GENERATED data
# -------------------------
print("[INFO] Loading generated CIFAR-10 images...")
gen_root = "generated_data(stable_diffusion)"

gen_images = []
gen_labels = []

class_map = {
    "generated_airplane": 0,
    "generated_automobile": 1,
    "generated_bird": 2,
    "generated_cat": 3,
    "generated_deer": 4,
    "generated_dog": 5,
    "generated_frog": 6,
    "generated_horse": 7,
    "generated_ship": 8,
    "generated_truck": 9
}

for folder, label in class_map.items():
    folder_path = os.path.join(gen_root, folder)
    if not os.path.exists(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        if img_name.endswith(".png") or img_name.endswith(".jpg"):
            img_path = os.path.join(folder_path, img_name)

            img = transforms.functional.to_tensor(
                plt.imread(img_path)
            )

            if img.shape[0] == 4:  # RGBA → RGB
                img = img[:3]

            img = img.unsqueeze(0)
            gen_images.append(img)
            gen_labels.append(label)

            if len(gen_images) >= MAX_SAMPLES:
                break

    if len(gen_images) >= MAX_SAMPLES:
        break

gen_images = torch.cat(gen_images).to(DEVICE)
gen_labels = np.array(gen_labels)

with torch.no_grad():
    gen_feat = victim.features(gen_images)
    gen_feat = gen_feat.view(gen_feat.size(0), -1).cpu().numpy()

# -------------------------
# PCA
# -------------------------
print("[INFO] Running PCA...")
pca = PCA(n_components=2)
real_pca = pca.fit_transform(real_feat)
gen_pca = pca.transform(gen_feat)

# -------------------------
# Plot
# -------------------------
print("[INFO] Saving PCA visualization...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(real_pca[:, 0], real_pca[:, 1], c=real_labels, s=5, cmap="tab10")
plt.title("Distribution of real training data")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.scatter(gen_pca[:, 0], gen_pca[:, 1], c=gen_labels, s=5, cmap="tab10")
plt.title("Distribution of generating data")
plt.axis("off")

plt.tight_layout()
plt.savefig("pca_distribution_cifar10.png", dpi=300)
plt.show()

print("[DONE] PCA image saved as pca_distribution_cifar10.png")
