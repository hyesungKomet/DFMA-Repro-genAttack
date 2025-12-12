import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils_cifar import *   # victim model definitions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# Helper: Show an image
# -------------------------------------------------------------------------
def imshow(img):
    img = img.detach().cpu()
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# -------------------------------------------------------------------------
# Extract VICTIM FEATURE (4096-D)
# -------------------------------------------------------------------------
def get_features(model, x):
    model.eval()

    with torch.no_grad():
        # forward until classifier input
        x = model.features(x)
        x = model.avgpool(x)
        x = x.view(x.size(0), -1)     # (B, 4096)
        x = model.classifier[0](x)    # first FC layer → still 4096
    return x


# -------------------------------------------------------------------------
# CIFAR-10 INVERSION NETWORK
# (Leader Requirement: 1 FC + 4 CNN/Deconv pairs)
# -------------------------------------------------------------------------
class CIFARInversionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # FC Layer: 4096 → 256 * 2 * 2
        self.fc = nn.Linear(4096, 256 * 2 * 2)

        # 4 blocks → (Conv → Deconv)
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 256, 2, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# -------------------------------------------------------------------------
# Load victim classifier
# -------------------------------------------------------------------------
print("[INFO] Loading victim model...")
victim = torch.load("vgg16_victim_cifar10_state.pt", map_location=torch.device("cpu"))
victim.to(device)
victim.eval()

# Confirm feature shape:
dummy = torch.randn(1, 3, 32, 32).to(device)
print("[DEBUG]  Feature shape:", get_features(victim, dummy).shape)   # Expect (1, 4096)


# -------------------------------------------------------------------------
# Load synthetic data from GAN
# -------------------------------------------------------------------------
print("[INFO] Loading synthetic generated CIFAR images...")
gen_img = torch.load("generated_data.pt").to(device)     # (N, 3, 32, 32)

# GET victim features for each image
print("[INFO] Extracting victim features...")
features = []

for i in range(0, len(gen_img), 64):
    batch = gen_img[i:i+64]
    f = get_features(victim, batch)
    features.append(f.cpu())

feat_tensor = torch.cat(features, dim=0)
print("[DEBUG] Feature tensor:", feat_tensor.shape)


dataset = Data.TensorDataset(gen_img.cpu(), feat_tensor)
loader = Data.DataLoader(dataset, batch_size=64, shuffle=True)


# -------------------------------------------------------------------------
# Train inversion model
# -------------------------------------------------------------------------
net = CIFARInversionNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0004)

print("[INFO] Starting inversion training...")

epochs = 40
for epoch in range(epochs):
    total_loss = 0

    for imgs, feats in loader:
        imgs = imgs.to(device)
        feats = feats.to(device)

        optimizer.zero_grad()
        recon = net(feats)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            print("[INFO] Showing sample reconstruction...")
            imshow(recon[0])


# -------------------------------------------------------------------------
# Evaluate inversion accuracy
# -------------------------------------------------------------------------
print("[INFO] Computing inversion accuracy...")

correct = 0
total = 0

with torch.no_grad():
    for imgs, feats in loader:
        imgs = imgs.to(device)
        feats = feats.to(device)

        recon = net(feats)

        pred_real = victim(imgs).argmax(1)
        pred_fake = victim(recon).argmax(1)

        correct += (pred_real == pred_fake).sum().item()
        total += len(imgs)

acc = correct / total
print(f"[RESULT] CIFAR Inversion Accuracy: {acc:.4f}")


torch.save(net.state_dict(), "cifar_inversion_model.pth")
print("[DONE] Saved inversion network.")
