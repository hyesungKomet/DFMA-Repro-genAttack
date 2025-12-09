import torch.utils.data
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

# -----------------------
#   Helper: Show Samples
# -----------------------
def show_generated_images(images):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

# -----------------------
#   Image Transform
# -----------------------
image_size = 64
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# -----------------------
#   Load Dataset
# -----------------------
full_train_dataset = dset.ImageFolder(root='./melanoma_cancer_dataset/train', transform=transform)

train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
_, gan_data = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(gan_data, batch_size=128, shuffle=True)

val_dataset = dset.ImageFolder(root='./melanoma_cancer_dataset/test', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------
#   Model Parameters
# -----------------------
nz = 100    # latent size
ngf = 64
ndf = 64
nc = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------
#   Generator Definition
# -----------------------
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# -----------------------
#   Fix for PyTorch 2.6 — Allow loading Generator()
# -----------------------
import torch.serialization
torch.serialization.add_safe_globals([Generator])

# -----------------------
#   Load Victim + GANs
# -----------------------
print("[INFO] Loading victim model + GAN generators...")

victim_model = torch.load('victim_model.pt', weights_only=False)
netG = torch.load('netG_malignant.pt', weights_only=False)
netG_B = torch.load('netG_benign.pt', weights_only=False)

victim_model = victim_model.to(device)
netG = netG.to(device)
netG_B = netG_B.to(device)

victim_model.eval()
netG.eval()
netG_B.eval()

# -----------------------
#   Generate Synthetic Data
# -----------------------
print("[INFO] Generating synthetic data...")

generate_img, generate_label = None, None

with torch.no_grad():
    for i in range(2000):

        # generate malignant & benign samples
        z_m = torch.randn(10, nz, 1, 1, device=device)
        z_b = torch.randn(10, nz, 1, 1, device=device)

        fake_malignant = netG(z_m)
        fake_benign = netG_B(z_b)

        # classify using victim model
        pred_m = victim_model(fake_malignant)
        pred_b = victim_model(fake_benign)

        # concatenate into dataset
        if generate_img is None:
            generate_img = torch.cat([fake_malignant, fake_benign], dim=0)
            generate_label = torch.cat([pred_m, pred_b], dim=0)
        else:
            generate_img = torch.cat([generate_img, fake_malignant, fake_benign], dim=0)
            generate_label = torch.cat([generate_label, pred_m, pred_b], dim=0)

        if i % 100 == 0:
            print(f"Generated {i * 20} samples...")

# -----------------------
#   Save Outputs
# -----------------------
torch.save(generate_img, 'generated_data.pt')
torch.save(generate_label, 'generated_label.pt')

print("[DONE] Saved generated_data.pt and generated_label.pt")
