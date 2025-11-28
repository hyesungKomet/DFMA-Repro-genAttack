import torch.utils.data
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def show_generated_images(images):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

image_size=64
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

full_train_dataset = dset.ImageFolder(root='./melanoma_cancer_dataset/train', transform=transform)
# 划分训练集和验证集
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
_, gan_data = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
trainloader = torch.utils.data.DataLoader(gan_data,batch_size=128,shuffle=True)
val_dataset = dset.ImageFolder(root='./melanoma_cancer_dataset/test', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# Parameters
nz = 100  # Size of generator input (latent vector)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
nc = 3    # Number of channels in the training images (for color images this is 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

model=torch.load('victim_model.pt')
netG=torch.load('netG_malignant.pt')
netG_B=torch.load('netG_benign.pt')
netG_B.to(device)
netG.to(device)
model.to(device)
model.eval()
netG.eval()
netG_B.eval()

generate_img,generate_label=None,None
with torch.no_grad():
    for i in range(2000):
        fixed_noise = torch.randn(10, nz, 1, 1, device=device)
        fix_noise = torch.randn(10, nz, 1, 1, device=device)
        fake_images = netG(fixed_noise)
        fake_images_benign=netG_B(fix_noise)
        output=model(fake_images)
        output_benign=model(fake_images_benign)
        if generate_img is not None:
            generate_img=torch.cat([generate_img,fake_images,fake_images_benign],dim=0)
            generate_label = torch.cat([generate_label, output,output_benign], dim=0)
        else:
            generate_img=fake_images
            generate_label=output

torch.save(generate_img,'generated_data.pt')
torch.save(generate_label,'generated_label.pt')