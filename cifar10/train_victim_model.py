# python train_victim_model.py --model resnet18

import os
from pathlib import Path
import csv
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_cifar import *

# Argument parser
parser = argparse.ArgumentParser(description="Train victim model (VGG19 or ResNet18) on CIFAR-10")
parser.add_argument('--model', type=str, default='vgg19', choices=['vgg19', 'resnet18'],
                    help='Model architecture to use (default: vgg19)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
args = parser.parse_args()

# Device detection: prefer CUDA, then MPS (Apple), then CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")
print(f"Model: {args.model}, Epochs: {args.epochs}, Batch size: {args.batch_size}")

# load data set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# load model based on argument and move to device
if args.model == 'vgg19':
    model = get_vggmodel().to(device)
elif args.model == 'resnet18':
    model = get_resnet18model().to(device)
else:
    raise ValueError(f"Unknown model: {args.model}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# results directory (organized by model name)
results_dir = Path("results") / args.model
results_dir.mkdir(parents=True, exist_ok=True)

# CSV log file
log_path = results_dir / f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "test_accuracy"])  # header

# training loop with tqdm
epochs = args.epochs
train_losses = []
test_accuracies = []
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}")
    for i, data in pbar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 50 == 0:
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)

    # test accuracy (utils_cifar now uses device-aware evaluation)
    accuracy = test_accuracy(model, testloader)
    test_accuracies.append(accuracy)

    # write log
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, epoch_loss, accuracy])

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# save model (state_dict) and full model for convenience
torch.save(model.state_dict(), results_dir / f'{args.model}_victim_cifar10_state.pt')
try:
    torch.save(model, results_dir / f'{args.model}_victim_cifar10_full.pt')
except Exception:
    # saving full model may fail for some setups; state_dict is primary
    pass

# plot losses and accuracies
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
plt.title(f'{args.model.upper()} - Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, marker='o', label='Test Accuracy', color='orange')
plt.title(f'{args.model.upper()} - Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / f'{args.model}_train_plots.png', dpi=100)
plt.close()

print(f'Finished Training [{args.model}] — artifacts saved in {results_dir}')