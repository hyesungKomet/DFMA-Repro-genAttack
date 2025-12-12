import os
from pathlib import Path
import csv
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_mnist import *

# Argument parser
parser = argparse.ArgumentParser(description="Train victim model (CNN) on MNIST")
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
args = parser.parse_args()

# Device detection: prefer CUDA, then MPS (Apple), then CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

torch.manual_seed(10)
# load data set mnist(0~9)
transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

# load model
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# results directory (organized by model name)
model_name = 'cnn'
results_dir = Path("results") / (model_name+'_victim')
results_dir.mkdir(parents=True, exist_ok=True)

# CSV log file
log_path = results_dir / f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "test_accuracy"])  # header

# start training
epochs = args.epochs
train_losses = []
test_accuracies = []
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}")
    for i, data in pbar:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)
    
    # test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    
    # write log
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, epoch_loss, accuracy])
    
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# save model (state_dict) and full model for convenience
torch.save(model.state_dict(), results_dir / f'{model_name}_victim_mnist_state.pth')
try:
    torch.save(model, results_dir / f'{model_name}_victim_mnist_full.pth')
except Exception:
    # saving full model may fail for some setups; state_dict is primary
    pass

# plot losses and accuracies
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
plt.title('CNN - Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, marker='o', label='Test Accuracy', color='orange')
plt.title('CNN - Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / f'{model_name}_train_plots.png', dpi=100)
plt.close()

print(f'Finished Training [CNN] — artifacts saved in {results_dir}')

