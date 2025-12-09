import torch
import torch.utils.data as Data
from utils_skin import *
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading victim model...")
victim = torch.load("victim_model.pt", weights_only=False)
victim.to(device)
victim.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("[INFO] Loading GAN-generated images...")
imgs = torch.load("generated_data.pt")     # (N, 3, 64, 64)
labels = torch.load("generated_label.pt")  # logits or soft labels

print("Dataset shape:", imgs.shape, labels.shape)

reverse_imgs = []
reverse_feats = []

print("[INFO] Extracting victim features...")

with torch.no_grad():
    for i in range(imgs.size(0)):
        img = imgs[i].unsqueeze(0).to(device)
        feat = victim.features(img)        # (1, 512, 1, 1)

        reverse_imgs.append(img.squeeze(0).cpu())
        reverse_feats.append(feat.squeeze().cpu())

        if i % 500 == 0:
            print(f"Processed {i}/{imgs.size(0)}")

reverse_imgs = torch.stack(reverse_imgs)
reverse_feats = torch.stack(reverse_feats)

print("[INFO] Saving reverse dataset...")
torch.save(reverse_imgs, "reverse_data.pt")
torch.save(reverse_feats, "reverse_label.pt")

print("[DONE] reverse_data.pt and reverse_label.pt created.")
