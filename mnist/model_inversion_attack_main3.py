import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.serialization

from utils_mnist import CNN   # allowlist this class for safe loading


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)


# -------------------------------------------------
# Safe Loading Function (patched)
# -------------------------------------------------
def load_victim_model(path):
    """
    Loads either:
    1) a full model saved with torch.save(model, path)
    2) a state_dict saved with torch.save(model.state_dict(), path)
    """

    print(f"[INFO] Loading victim model: {path}")

    # allow CNN class during loading
    torch.serialization.add_safe_globals([CNN])

    try:
        # Try loading as full model first
        model = torch.load(path, map_location="cpu", weights_only=False)
        print("[INFO] Loaded FULL MODEL successfully.")
        return model

    except Exception as e1:
        print("[WARN] Full model load failed:", e1)
        print("[INFO] Trying to load as state_dict...")

        # Try loading as state_dict
        model = CNN()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print("[INFO] Loaded STATE_DICT successfully.")
        return model


# -------------------------------------------------
# Load MNIST
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)


# -------------------------------------------------
# Load victim classifier (patched)
# -------------------------------------------------
victim = load_victim_model("cnn_victim_mnist_full.pth")
victim = victim.to(device)
victim.eval()


# -------------------------------------------------
# Extract intermediate features for inversion
# -------------------------------------------------
def extract_features(model, x):
    x = model.conv1(x)
    x = torch.relu(x)
    x = torch.max_pool2d(x, 2)

    x = model.conv2(x)
    x = torch.relu(x)
    x = torch.max_pool2d(x, 2)

    x = x.view(-1, 64 * 5 * 5)
    x = model.fc1(x)
    x = torch.relu(x)

    return x      # (batch, 128)


# -------------------------------------------------
# Load generated synthetic data
# -------------------------------------------------
gen_img = torch.load("generated_data.pt").to(device)          # images
gen_logits = torch.load("generated_label.pt").to(device)      # soft labels

# Convert synthetic logits -> correct victim feature targets
gen_features = extract_features(victim, gen_img).detach()

dataset = Data.TensorDataset(gen_features, gen_img)
loader = Data.DataLoader(dataset, batch_size=64, shuffle=True)


# -------------------------------------------------
# Inversion Network: 1 FC + 3 Deconv blocks
# -------------------------------------------------
class InversionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(128, 128 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),

            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x


model = InversionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# -------------------------------------------------
# Train inversion model
# -------------------------------------------------
print("[INFO] Training inversion network...")

for epoch in range(40):
    total_loss = 0
    for feat, img in loader:
        optimizer.zero_grad()
        out = model(feat)
        loss = criterion(out, img)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/40  Loss: {total_loss/len(loader):.4f}")


# -------------------------------------------------
# Evaluation — MSE + Victim-Class Accuracy
# -------------------------------------------------
print("[INFO] Evaluating...")

demo_imgs = []
demo_labels = []
for i, (img, lab) in enumerate(testset):
    if i % 10 == 0:
        demo_imgs.append(img)
        demo_labels.append(lab)

correct = 0
total_mse = 0

with torch.no_grad():
    for img, label in zip(demo_imgs, demo_labels):
        img = img.to(device)

        feat = extract_features(victim, img.unsqueeze(0))
        rec = model(feat)

        total_mse += criterion(rec[0], img).item()

        pred = victim(rec).argmax()
        if pred.item() == label:
            correct += 1

accuracy = correct / len(demo_imgs)

print(f"[RESULT] MSE:      {total_mse / len(demo_imgs):.4f}")
print(f"[RESULT] Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "inversion_mnist.pth")
print("[DONE] Saved inversion model.")
