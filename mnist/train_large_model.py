import time
import random
import torchvision
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data

from utils_mnist import *  # CNN, Generator, Discriminator, ngpu 등

# Device 설정 (CUDA -> MPS -> CPU)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.manual_seed(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# -----------------------------
# Hyperparameters
# -----------------------------
dataroot = "data"
workers = 0
batch_size = 100
image_size = 28
nc = 1
num_classes = 10
nz = 100
ngf = 64
ndf = 64
num_epochs = 10
lr = 0.0003
beta1 = 0.5
# ngpu는 utils_mnist에 이미 있으니 그대로 사용 (필요하면 여기서 덮어써도 됨)
# ngpu = 1

# -----------------------------
# 데이터 준비 (EMNIST + MNIST)
# -----------------------------
my_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.EMNIST(
    root=dataroot,
    split='digits',
    train=True,
    transform=my_transform,
    download=True
)
test_data = datasets.EMNIST(
    root=dataroot,
    split='digits',
    train=False,
    transform=my_transform
)

my_data, mylabel = [], []
for idx, (data, target) in enumerate(train_data):
    if idx > 99999:
        break
    # EMNIST 숫자 회전/플립 보정
    data[0] = torch.rot90(torch.flipud(data[0]), k=-1)
    my_data.append(data)
    mylabel.append(target)

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=my_transform
)

# select the size of subset
subset_size = 20000
total_samples = len(trainset)
subset_indices = random.sample(range(total_samples), subset_size)
subset_dataset = torch.utils.data.Subset(trainset, subset_indices)

for i, (data, target) in enumerate(subset_dataset):
    my_data.append(data)
    mylabel.append(target)
    if i > 20000:
        break

mylabel = torch.LongTensor(mylabel)
my_data = torch.FloatTensor(torch.stack(my_data))
torch_dataset = Data.TensorDataset(my_data, mylabel)
print(f'Total Size of Dataset: {len(torch_dataset)}')

dataloader = DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)

# -----------------------------
# 모델 정의 & 초기화
# -----------------------------
# Generator
netG = Generator(ngpu).to(device)
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

# Discriminator
netD = Discriminator(ngpu).to(device)
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

criterion = nn.BCELoss()

real_label_num = 1.0
fake_label_num = 0.0

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Label one-hot for G
label_1hots = torch.zeros(10, 10)
for i in range(10):
    label_1hots[i, i] = 1
label_1hots = label_1hots.view(10, 10, 1, 1).to(device)

# Label one-hot for D (image_size x image_size mask)
label_fills = torch.zeros(10, 10, image_size, image_size)
ones = torch.ones(image_size, image_size)
for i in range(10):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)

# -----------------------------
# 학습 루프 준비
# -----------------------------
img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10.0

print("Starting Training Loop...")
# -----------------------------
# 학습 루프
# -----------------------------
for epoch in range(num_epochs):
    beg_time = time.time()
    netG.train()
    netG.to(device)
    netD.train()
    netD.to(device)

    for i, data in enumerate(dataloader):
        # fixed_noise / fixed_label는 에폭 중간에도 그대로 재사용 가능
        fixed_noise = torch.randn(100, nz, 1, 1, device=device)
        fixed_label = label_1hots[torch.arange(10).repeat(10).sort().values]

        # -----------------
        # (1) Discriminator 업데이트
        # -----------------
        netD.zero_grad()

        real_image = data[0].to(device)
        b_size = real_image.size(0)
        real_label = torch.full((b_size,), real_label_num, device=device)
        fake_label = torch.full((b_size,), fake_label_num, device=device)

        G_label = label_1hots[data[1]].to(device)
        D_label = label_fills[data[1]].to(device)

        # Real batch
        output = netD(real_image, D_label).view(-1)
        errD_real = criterion(output, real_label)
        errD_real.backward()
        D_x = output.mean().item()

        # Fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise, G_label)
        output = netD(fake.detach(), D_label).view(-1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        # -----------------
        # (2) Generator 업데이트
        # -----------------
        netG.zero_grad()
        output = netD(fake, D_label).view(-1)
        errG = criterion(output, real_label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        end_time = time.time()
        run_time = round(end_time - beg_time)
        print(
            f'Epoch: [{epoch + 1:0>{len(str(num_epochs))}}/{num_epochs}]',
            f'Step: [{i + 1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
            f'Loss-D: {errD.item():.4f}',
            f'Loss-G: {errG.item():.4f}',
            f'D(x): {D_x:.4f}',
            f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
            f'Time: {run_time}s',
            end='\r'
        )

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_x_list.append(D_x)
        D_z_list.append(D_G_z2)

        # Best G 저장
        if errG.item() < loss_tep:
            torch.save(netG.state_dict(), 'model.pt')
            loss_tep = errG.item()

    print()  # 줄바꿈

    # -----------------
    # 에폭별 샘플 이미지 시각화
    # -----------------
    netG.eval()

    # 시각화는 CPU에서
    netG_cpu = netG.to("cpu")
    with torch.no_grad():
        fake = netG_cpu(fixed_noise.cpu(), fixed_label.cpu()).detach()
    img = utils.make_grid(fake, nrow=10)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"Fake Images (Epoch {epoch+1})")
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()

    img_list.append(img)

    # 다음 epoch을 위해 다시 device로
    netG.to(device)

# 최종 G 저장
torch.save(netG.state_dict(), 'last_model.pt')
print("Training finished. Saved generator to last_model.pt")