import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.datasets import ImageFolder
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# --------------------------------------------------
# CIFAR-10 class names
# --------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# Device
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load victim model (FULL MODEL CHECKPOINT)
# --------------------------------------------------
print("[INFO] Loading victim model...")

victim = torch.load(
    "victim_model.pt",
    map_location=DEVICE,
    weights_only=False  # PyTorch 2.6+ fix
)

victim = victim.to(DEVICE)
victim.eval()

# --------------------------------------------------
# Feature extractor (VGG feature space)
# --------------------------------------------------
def extract_features(model, x):
    with torch.no_grad():
        feat = model.features(x)
        feat = torch.flatten(feat, 1)
    return feat

# --------------------------------------------------
# Transforms (ImageNet normalization for VGG)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# --------------------------------------------------
# Load REAL CIFAR-10
# --------------------------------------------------
print("[INFO] Loading real CIFAR-10 data...")

real_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

real_loader = torch.utils.data.DataLoader(
    real_dataset,
    batch_size=64,
    shuffle=False
)

real_features, real_labels = [], []

for imgs, labels in real_loader:
    imgs = imgs.to(DEVICE)
    feats = extract_features(victim, imgs)

    real_features.append(feats.cpu())
    real_labels.append(labels)

real_features = torch.cat(real_features)
real_labels = torch.cat(real_labels)

print("[DEBUG] Real feature shape:", real_features.shape)

# --------------------------------------------------
# Load GENERATED CIFAR-10 (ImageFolder)
# Folder structure:
# generated_data (stable_diffusion)/
# ├── airplane/
# ├── automobile/
# ├── ...
# └── truck/
# --------------------------------------------------
print("[INFO] Loading generated CIFAR-10 data...")

gen_dataset = ImageFolder(
    root="generated_data (stable_diffusion)",
    transform=transform
)

gen_loader = torch.utils.data.DataLoader(
    gen_dataset,
    batch_size=64,
    shuffle=False
)

gen_features, gen_labels = [], []

for imgs, labels in gen_loader:
    imgs = imgs.to(DEVICE)
    feats = extract_features(victim, imgs)

    gen_features.append(feats.cpu())
    gen_labels.append(labels)

gen_features = torch.cat(gen_features)
gen_labels = torch.cat(gen_labels)

print("[DEBUG] Generated feature shape:", gen_features.shape)

# --------------------------------------------------
# PCA
# --------------------------------------------------
print("[INFO] Running PCA...")

all_features = torch.cat([real_features, gen_features], dim=0).numpy()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_features)

real_pca = pca_result[:len(real_features)]
gen_pca  = pca_result[len(real_features):]

# --------------------------------------------------
# Plot PCA (class-colored, paper-ready)
# --------------------------------------------------
print("[INFO] Plotting PCA...")

plt.figure(figsize=(10, 8))
cmap = plt.get_cmap("tab10")

# --- Real data (circles) ---
for cls in range(10):
    idx = (real_labels == cls)
    plt.scatter(
        real_pca[idx, 0],
        real_pca[idx, 1],
        color=cmap(cls),
        alpha=0.35,
        s=8,
        marker="o"
    )

# --- Generated data (crosses) ---
for cls in range(10):
    idx = (gen_labels == cls)
    plt.scatter(
        gen_pca[idx, 0],
        gen_pca[idx, 1],
        color=cmap(cls),
        alpha=0.7,
        s=12,
        marker="x"
    )

# --- Class legend ---
class_legend = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=CIFAR10_CLASSES[i],
        markerfacecolor=cmap(i),
        markersize=8
    )
    for i in range(10)
]

plt.legend(
    handles=class_legend,
    title="CIFAR-10 Classes",
    loc="upper right"
)

plt.title("PCA of Victim Model Feature Space (CIFAR-10)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("pca_distribution_cifar10.png", dpi=300)
plt.show()

print("[DONE] Saved PCA as pca_distribution_cifar10.png")
