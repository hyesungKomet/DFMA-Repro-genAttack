import os
from pathlib import Path
import csv
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_skin import *

# Argument parser
parser = argparse.ArgumentParser(description="Train victim model (VGG16) on Skin Cancer")
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
args = parser.parse_args()

# Device detection: prefer CUDA, then MPS (Apple), then CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

# Some MPS builds do not support certain adaptive pooling ops used by VGG.
# If running on MPS, force computation to CPU to avoid runtime errors.
if device.type == 'mps':
    print("WARNING: MPS backend may not support some ops used by this model. Using CPU for computation.")
    compute_device = torch.device('cpu')
else:
    compute_device = device

# load training data set
transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
full_train_dataset = ImageFolder(root='./melanoma_cancer_dataset/train', transform=transform)
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
val_dataset = ImageFolder(root='./melanoma_cancer_dataset/test', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
print(f"Class to Index: {val_dataset.class_to_idx}")

# create a model
num_epochs = args.epochs
model = get_vggmodel()
model.to(compute_device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# results directory (organized by model name)
model_name = 'vgg16_victim'
results_dir = Path("results") / model_name
results_dir.mkdir(parents=True, exist_ok=True)

# CSV log file
log_path = results_dir / f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])  # header

# train the model
train_losses = []
val_losses = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(compute_device), labels.to(compute_device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
    
    epoch_train_loss = total_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(compute_device), labels.to(compute_device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_accuracy)
    
    # write log
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, val_accuracy])
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# save model (state_dict) and full model for convenience
torch.save(model.state_dict(), results_dir / f'{model_name}_victim_skin_state.pt')
try:
    torch.save(model, results_dir / f'{model_name}_victim_skin_full.pt')
except Exception:
    # saving full model may fail for some setups; state_dict is primary
    pass

# plot losses and accuracies
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='s', label='Val Loss')
plt.title('VGG16 - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', label='Val Accuracy', color='orange')
plt.title('VGG16 - Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / f'{model_name}_train_plots.png', dpi=100)
plt.close()

print(f'Finished Training [VGG16] — artifacts saved in {results_dir}')
print("Training complete!")


