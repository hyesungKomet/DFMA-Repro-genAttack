import torch.utils.data
import torchvision.datasets as dset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm
import csv, os


def show_generated_images(images):
    pass
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)
        )
    )
    plt.show()


# -----------------------------
# Data & Transform
# -----------------------------
image_size = 64
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# benign 학습 시에는 ./gen_benign 으로 바꾸면 됨
full_train_dataset = dset.ImageFolder(root='./gen_benign', transform=transform)
# full_train_dataset = dset.ImageFolder(root="./gen_malignant", transform=transform)

# 70% / 30% split 후 여기서는 val 파트(gan_data)만 GAN 학습에 사용
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
_, gan_data = torch.utils.data.random_split(
    full_train_dataset, [train_size, val_size]
)
trainloader = torch.utils.data.DataLoader(
    gan_data, batch_size=128, shuffle=True
)

# -----------------------------
# Hyper-parameters
# -----------------------------
nz = 100  # latent vector size (input to G)
ngf = 64  # feature maps in G
ndf = 64  # feature maps in D
nc = 3  # RGB
batch_size = 128
learning_rate = 0.0002
num_epochs = 100
real_label = 1.0
fake_label = 0.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Models
# -----------------------------
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input Z goes into a convolution
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
            nn.Tanh(),  # [-1, 1]
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: (nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # (ndf, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # (ndf*2, 16, 16)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # (ndf*2 * 16 * 16) = 128*16*16
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        conv_output = self.main(input)
        fc_output = self.fc_layers(conv_output)
        return fc_output


netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(ndf, nc).to(device)

# -----------------------------
# Loss & Optimizers
# -----------------------------
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# fixed noise for monitoring generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# For logging
epoch_D_losses = []
epoch_G_losses = []

# -----------------------------
# Training Loop
# -----------------------------
os.makedirs("generated_samples", exist_ok=True)
for epoch in range(num_epochs):
    running_D_loss = 0.0
    running_G_loss = 0.0

    pbar = tqdm(
        enumerate(trainloader, 0),
        total=len(trainloader),
        desc=f"Epoch {epoch + 1}/{num_epochs}",
    )

    for i, (real_images, _) in pbar:
        ############################
        # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()

        # Train with real batch
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        label = torch.full(
            (b_size,), real_label, dtype=torch.float, device=device
        )

        output = netD(real_images).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # G wants D(G(z)) to be real
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # accumulate losses
        running_D_loss += errD.item()
        running_G_loss += errG.item()

        # update tqdm postfix
        pbar.set_postfix(
            {
                "Loss_D": f"{errD.item():.4f}",
                "Loss_G": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.4f}",
                "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}",
            }
        )

        # occasionally print & visualize
        if i % 200 == 0:
            with torch.no_grad():
                fake_vis = netG(fixed_noise).detach().cpu()
            save_path = f"generated_samples/epoch_{epoch+1}_step_{i}.png"
            vutils.save_image(fake_vis, save_path, normalize=True, nrow=8)
            print(f"[INFO] Saved generated image to {save_path}")

    # end of epoch: avg losses
    avg_D = running_D_loss / len(trainloader)
    avg_G = running_G_loss / len(trainloader)
    epoch_D_losses.append(avg_D)
    epoch_G_losses.append(avg_G)
    print(
        f"[Epoch {epoch + 1}/{num_epochs}] Avg Loss_D: {avg_D:.4f} | Avg Loss_G: {avg_G:.4f}"
    )

# -----------------------------
# Save generator
# -----------------------------
# torch.save(netG, "netG_malignant.pt")
# benign 학습 버전일 때는 아래처럼:
torch.save(netG, 'netG_benign.pt')

# -----------------------------
# Save training log as CSV
# -----------------------------
with open("gan_train_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "avg_loss_D", "avg_loss_G"])
    for e, (d, g) in enumerate(zip(epoch_D_losses, epoch_G_losses), start=1):
        writer.writerow([e, d, g])

print("Training log saved to gan_train_log.csv")
print("Training finished. Generator saved to netG_malignant.pt")