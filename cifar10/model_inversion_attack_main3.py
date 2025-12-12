import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.models import vgg16

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Utility: show image
# ---------------------------
def imshow(img):
    img = img.detach().cpu()
    img = (img + 1) / 2
    npimg = img.numpy().transpose(1, 2, 0)
    plt.imshow(npimg)
    plt.axis("off")
    plt.show()


# ---------------------------
# Victim Feature Extractor
# ---------------------------
def get_features(model, x):
    with torch.no_grad():
        feat = model.features(x)
        feat = model.avgpool(feat)
        feat = feat.view(feat.size(0), -1)    # 25088

        feat = model.classifier[0](feat)      # fc1 → 4096
        feat = model.classifier[1](feat)      # ReLU
    return feat                           # (B, 4096)



# ============================================================
# CIFAR-10 INVERSION NETWORK
# Requirement: 1 FC + 4 (Conv + TransposedConv) Pairs
# ============================================================
class CIFARInversionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Step 1: FC layer: 4096 -> (256 × 2 × 2)
        self.fc = nn.Linear(4096, 256 * 2 * 2)

        # Pair 1: (2→4)
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU()
        )

        # Pair 2: (4→8)
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )

        # Pair 3: (8→16)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU()
        )

        # Pair 4: (16→32)
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()          # output [-1,1]
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 256, 2, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

print("[INFO] Loading victim model...")

victim = vgg16(weights=None)
victim.classifier[6] = nn.Linear(4096, 10)     # CIFAR10: 10 classes

state_dict = torch.load("vgg16_victim_cifar10_state.pt", map_location="cpu")
victim.load_state_dict(state_dict)

victim.to(device)
victim.eval()

# Check feature size
dummy = torch.randn(1, 3, 32, 32).to(device)
print("[DEBUG] Feature shape:", get_features(victim, dummy).shape)


# ============================================================
# Load synthetic GAN data
# ============================================================
print("[INFO] Loading synthetic generated CIFAR images...")

gen_img = torch.load("generated_data.pt")   # Must be (N, 3, 32, 32)
gen_img = gen_img.to(device)

# Extract victim features
print("[INFO] Extracting victim features...")
all_feat = []

batch_size = 64
for i in range(0, len(gen_img), batch_size):
    batch = gen_img[i:i+batch_size]
    feat = get_features(victim, batch)
    all_feat.append(feat.cpu())

features = torch.cat(all_feat, dim=0)       # (N, 4096)
print("[DEBUG] Feature tensor:", features.shape)


# Build dataset
dataset = Data.TensorDataset(gen_img.cpu(), features)
loader = Data.DataLoader(dataset, batch_size=64, shuffle=True)


# ============================================================
# Init Inversion Net
# ============================================================
net = CIFARInversionNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

print("[INFO] Starting inversion training...")


# ============================================================
# Training Loop
# ============================================================
epochs = 40
for epoch in range(epochs):
    running_loss = 0

    for img, feat in loader:
        img, feat = img.to(device), feat.to(device)

        optimizer.zero_grad()
        recon = net(feat)
        loss = criterion(recon, img)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

    # Show example every 10 epochs
    if (epoch + 1) % 10 == 0:
        print("[INFO] Showing sample reconstruction...")
        imshow(recon[0])


# ============================================================
# Evaluation Metric
# ============================================================
print("[INFO] Computing inversion accuracy...")

correct = 0
total = 0

with torch.no_grad():
    for img, feat in loader:
        img = img.to(device)
        feat = feat.to(device)

        recon = net(feat)
        real_pred = victim(img)
        recon_pred = victim(recon)

        correct += (real_pred.argmax(1) == recon_pred.argmax(1)).sum().item()
        total += img.size(0)

acc = correct / total
print(f"[RESULT] CIFAR Inversion Accuracy: {acc:.4f}")


# ============================================================
# Save Model
# ============================================================
torch.save(net.state_dict(), "cifar_inversion_model.pth")
print("[DONE] Saved inversion model.")
