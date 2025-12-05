import os
from pathlib import Path
import csv
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_cifar import *

# Argument parser
parser = argparse.ArgumentParser(description="Model Extraction Attack on CIFAR-10")
parser.add_argument('--model', type=str, default='vgg16', choices=['resnet18', 'vgg16'],
                    help='Extracted model architecture to use (default: vgg16)')
parser.add_argument('--victim-model', type=str, default='vgg16', choices=['resnet18', 'vgg16'],
                    help='Extracted model architecture to use (default: vgg16)')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--gen-model', type=str, default='SD1.5', help='Name of generative model used (e.g., SDXL, SD1.5)')
parser.add_argument('--augment', action='store_true', help='Whether generated data used augmentation')
parser.add_argument('--patience', type=int, default=7,
                    help='Early stopping patience (epochs) based on agreement metric.')
args = parser.parse_args()

# Device detection: prefer CUDA, then MPS (Apple), then CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")
print(f"Extracted Model: {args.model}, Victim Model: {args.victim_model}")
print(f"Generative model: {args.gen_model}, Augment: {args.augment}")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

# Load extracted model based on argument
if args.model == 'vgg16':
    extracted_model = get_vggmodel().to(device)
elif args.model == 'resnet18':
    extracted_model = get_resnet18model().to(device)
else:
    raise ValueError(f"Unknown extracted model: {args.model}")

transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# Load victim model based on argument
if args.victim_model == 'vgg16':
    victim_path = './results/vgg16_victim/vgg16_victim_cifar10_state.pt'
    victim_builder = get_vggmodel
elif args.victim_model == 'resnet18':
    victim_path = './results/resnet18_victim/resnet18_victim_cifar10_state.pt'
    victim_builder = get_resnet18model
else:
    raise ValueError(f"Unknown victim model: {args.victim_model}")

victim_obj = torch.load(victim_path, map_location=device)
if isinstance(victim_obj, dict):
    victim_model = victim_builder()
    victim_model.load_state_dict(victim_obj)
else:
    victim_model = victim_obj

victim_model = victim_model.to(device)
victim_model.eval()

# Load dataset from generated_data_{gen_model} directory
gen_data_dir = Path(f'./stealing_set') / f'{args.gen_model}' / f"{args.model}_{args.augment}" 
if not gen_data_dir.exists():
    raise FileNotFoundError(f"Generated data directory not found: {gen_data_dir}")

if args.augment:
    data_path = gen_data_dir / 'generated_data.pt'
    label_path = gen_data_dir / 'generated_label.pt'
else:
    data_path = gen_data_dir / 'augmented_data.pt'
    label_path = gen_data_dir / 'augmented_label.pt'

if not data_path.exists() or not label_path.exists():
    raise FileNotFoundError(f"Generated data or label file not found in {gen_data_dir}")

new_data = torch.load(data_path, map_location=device)
new_label = torch.load(label_path, map_location=device)
print(f"Generated data shape: {new_data.shape}, labels shape: {new_label.shape}")

# Detect label format and choose appropriate loss
# If labels are one-hot / probability vectors (dim>1 and second dim >1) -> use MSE on softmax outputs
# If labels are class indices (dim==1) -> use CrossEntropyLoss
# use_mse_loss = False
# if new_label.dim() > 1 and new_label.shape[1] > 1:
#     use_mse_loss = True
use_mse_loss = (new_label.dim() > 1 and new_label.shape[1] > 1)

if use_mse_loss:
    # ensure labels are float tensors in same device
    new_label = new_label.to(device).float()
    criterion = nn.MSELoss()
    print("Using MSE loss (soft/one-hot labels).")
else:
    # convert labels to long class indices for CrossEntropyLoss
    new_label = new_label.to(device).long()
    criterion = nn.CrossEntropyLoss()
    print("Using CrossEntropyLoss (class index labels).")

new_dataset = Data.TensorDataset(new_data, new_label)
new_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)

optimizer = optim.SGD(extracted_model.parameters(), lr=0.001, momentum=0.9)

# results directory with model names
model_dir = f"{args.model}_extraction_{args.gen_model}_{args.augment}"
results_dir = Path("results") / model_dir
results_dir.mkdir(parents=True, exist_ok=True)

# CSV log file
log_path = results_dir / f"extraction_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "test_accuracy", "agreement_accuracy"])  # header

# Training loop
train_losses = []
test_accuracies = []
agreement_accuracies = []

# Early stopping variables
best_test_acc = -1.0
best_agreement = -1.0
best_epoch = -1
patience_counter = 0

monitor_metric = 'agreement'  # or 'accuracy'

for epoch in range(args.epochs):
    running_loss = 0.0
    extracted_model.train()
    pbar = tqdm(enumerate(new_loader, 0), total=len(new_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
    for i, data in pbar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = extracted_model(inputs)
        # If using MSE (soft targets), compare probabilities, otherwise use CrossEntropy with logits
        if use_mse_loss:
            # probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
        else:
            # labels are long class indices
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(new_loader)
    train_losses.append(epoch_loss)
    
    # Test accuracy
    extracted_model.eval()
    accuracy = test_accuracy(extracted_model, testloader)
    test_accuracies.append(accuracy)
    
    # Agreement accuracy
    extracted_model.eval()
    victim_model.eval()
    agreement_correct = 0
    agreement_total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            my_outputs = extracted_model(images)
            victim_outputs = victim_model(images)
            _, my_predicted = torch.max(my_outputs.data, 1)
            _, victim_predicted = torch.max(victim_outputs.data, 1)
            agreement_total += labels.size(0)
            agreement_correct += (my_predicted == victim_predicted).sum().item()
    agreement_accuracy = 100 * agreement_correct / agreement_total
    agreement_accuracies.append(agreement_accuracy)
    
    # Write log
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, epoch_loss, accuracy, agreement_accuracy])
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Agreement: {agreement_accuracy:.2f}%")

    # ======== Early stopping & best model tracking ========
    # metric
    if monitor_metric == 'agreement':
        current_metric = agreement_accuracy
        best_metric_so_far = best_agreement
        metric_name = "Agreement"
    else:
        current_metric = accuracy
        best_metric_so_far = best_test_acc
        metric_name = "Test Accuracy"

    # save best model if improved
    if current_metric > best_metric_so_far:
        best_epoch = epoch + 1
        patience_counter = 0

        if monitor_metric == 'agreement':
            # agreement based best renewal
            best_agreement = current_metric
            # accuracy renewal
            if accuracy > best_test_acc:
                best_test_acc = accuracy
        else:
            # accuracy based best renewal
            best_test_acc = current_metric
            if agreement_accuracy > best_agreement:
                best_agreement = agreement_accuracy

        # state_dict + full model
        torch.save(extracted_model.state_dict(), results_dir / f'{args.model}_extracted_cifar10_state.pt')
        try:
            torch.save(extracted_model, results_dir / f'{args.model}_extracted_cifar10_full.pt')
        except Exception:
            pass

        print(f"[Best updated] Epoch {best_epoch}: {metric_name} = {current_metric:.2f}% "
              f"(Best Test Acc: {best_test_acc:.2f}%, Best Agreement: {best_agreement:.2f}%)")
    else:
        patience_counter += 1
        print(f"[No improvement] {metric_name} did not improve for {patience_counter} epoch(s).")

        if patience_counter >= args.patience:
            print(f"[Early Stopping] Stop at epoch {epoch+1}. "
                  f"Best epoch: {best_epoch} "
                  f"(Test Acc: {best_test_acc:.2f}%, Best Agreement: {best_agreement:.2f}%)")
            break

# Save extracted model
# torch.save(extracted_model.state_dict(), results_dir / f'{args.model}_extracted_cifar10_state.pt')
# try:
#     torch.save(extracted_model, results_dir / f'{args.model}_extracted_cifar10_full.pt')
# except Exception:
#     pass

summary_path = results_dir / 'best_summary.txt'
with open(summary_path, 'w') as f:
    f.write(
        f"Best epoch: {best_epoch}\n"
        f"Best test accuracy: {best_test_acc:.4f}\n"
        f"Best agreement: {best_agreement:.4f}\n"
        f"Monitor metric: {monitor_metric}\n"
        f"Patience: {args.patience}\n"
    )

print(f"Best model summary saved to {summary_path}")

num_epochs_run = len(train_losses)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs_run + 1), train_losses, marker='o', label='Train Loss')
plt.title('Model Extraction - Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs_run + 1), test_accuracies, marker='o', label='Test Accuracy', color='orange')
plt.title('Model Extraction - Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(range(1, num_epochs_run + 1), agreement_accuracies, marker='s', label='Agreement', color='green')
plt.title('Model Extraction - Agreement with Victim')
plt.xlabel('Epoch')
plt.ylabel('Agreement (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'extraction_plots.png', dpi=100)
plt.close()

print(f'Finished Model Extraction Attack — artifacts saved in {results_dir}')
