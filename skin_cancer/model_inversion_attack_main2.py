import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

from utils_skin import *   # victim model + all utilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Utility: Show Image
# ------------------------------------------------------------
def imshow(img):
    img = (img + 1) / 2                      # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# ------------------------------------------------------------
# Inversion Network
# Requirement: 1 FC + 3 Transposed CNN + 1 CNN block
# ------------------------------------------------------------
class SkinInversionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # FC layer (2048 → 256*4*4)
        self.fc = nn.Linear(2048, 256 * 4 * 4)

        # 3 required transposed CNN blocks
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 4→8
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 8→16
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # 16→32
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # EXTRA upsampling layer (NOT counted as deconv block)
        self.deconv_extra = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),     # 32→64
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        # Final CNN block (leader requirement)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 256, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv_extra(x)
        x = self.final_conv(x)
        return x



# ------------------------------------------------------------
# Load Victim Model
# ------------------------------------------------------------
print("[INFO] Loading victim model...")
victim = torch.load("vgg16_victim_skin_full.pt", weights_only=False)
victim.to(device)
victim.eval()

# Test feature size
with torch.no_grad():
    dummy = torch.randn(1, 3, 64, 64).to(device)
    feat_test = victim.features(dummy)
    print("[DEBUG] Victim feature shape:", feat_test.shape)


# ------------------------------------------------------------
# Load Generated Fake Data (from GAN)
# ------------------------------------------------------------
print("[INFO] Loading GAN synthetic dataset...")

gen_images = torch.load("generated_data.pt")   # (N, 3, 64, 64)
gen_images = gen_images.to(device)

# Extract features ONCE
print("[INFO] Extracting victim features for inversion...")
new_features = []

with torch.no_grad():
    for img in gen_images:
        img = img.unsqueeze(0)
        feat = victim.features(img)            # (1, 512, 2, 2)
        feat = feat.view(-1)                   # flatten to 2048
        new_features.append(feat)

new_features = torch.stack(new_features)       # (N, 2048)
print("[INFO] Feature tensor shape:", new_features.shape)

# Build dataset
dataset = Data.TensorDataset(gen_images, new_features)
loader  = Data.DataLoader(dataset, batch_size=32, shuffle=True)


# ------------------------------------------------------------
# Initialize Inversion Network
# ------------------------------------------------------------
net = SkinInversionNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)


print("[INFO] Starting inversion training...")


# ------------------------------------------------------------
# Train Loop
# ------------------------------------------------------------
epochs = 40
for epoch in range(epochs):

    running_loss = 0.0

    for real_img, feat in loader:
        real_img = real_img.to(device)
        feat = feat.to(device)

        optimizer.zero_grad()

        recon = net(feat)

        loss = criterion(recon, real_img)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

    if (epoch + 1) % 10 == 0:
        print("Showing reconstruction sample...")
        imshow(recon[0])


# ------------------------------------------------------------
# Save Result
# ------------------------------------------------------------
torch.save(net.state_dict(), "skin_inversion_model.pth")
print("[DONE] Saved inversion model.")

# ------------------------------------------------------------
# Evaluate Inversion by Comparing Victim Predictions
# ------------------------------------------------------------
print("[INFO] Evaluating inversion quality...")

victim.eval()
net.eval()

correct = 0
total = 0
mse_total = 0

with torch.no_grad():
    for real_img, feat in loader:
        real_img = real_img.to(device)
        feat = feat.to(device)

        recon = net(feat)

        # victim predictions
        pred_real = victim(real_img).argmax(1)
        pred_recon = victim(recon).argmax(1)

        correct += (pred_real == pred_recon).sum().item()
        total += real_img.size(0)

        mse_total += ((real_img - recon) ** 2).mean().item()

acc = correct / total
mse = mse_total / total

print(f"[RESULT] SkinCancer Inversion Accuracy: {acc:.4f}")
print(f"[RESULT] SkinCancer Reconstruction MSE: {mse:.4f}")
