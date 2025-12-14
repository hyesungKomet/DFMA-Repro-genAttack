import argparse
import csv
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
from utils_mnist import *
import numpy as np
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(123)

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Model Extraction Attack on MNIST")
parser.add_argument('--victim-path', type=str, default='./results/cnn/cnn_victim_mnist_state.pth', help='Path to victim model (state_dict or full model)')
parser.add_argument('--generated-data', type=str, default='./stealing_set_GAN/generated_data.pt', help='Path to generated data (.pt)')
parser.add_argument('--generated-label', type=str, default='./stealing_set_GAN/generated_label.pt', help='Path to generated labels (.pt)')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
parser.add_argument('--test-batch-size', type=int, default=100, help='Test batch size')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--results-dir', type=str, default=None, help='Optional results directory')
parser.add_argument('--log-interval', type=int, default=100, help='Batches between log updates')
args = parser.parse_args()

# Device detection (prefer CUDA -> MPS -> CPU)
device = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

workers = 0

# Load test dataset (we use generated data for training)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)

# Load victim model (support state_dict or full model)
try:
    victim_obj = torch.load(args.victim_path, map_location='cpu')
    if isinstance(victim_obj, dict):
        victim_model = CNN()
        victim_model.load_state_dict(victim_obj)
    else:
        victim_model = victim_obj
except Exception:
    # try direct load
    victim_model = torch.load(args.victim_path, map_location='cpu')
victim_model = victim_model.to(device)
victim_model.eval()

# Load generated dataset
new_data = torch.load(args.generated_data, map_location='cpu')
new_label = torch.load(args.generated_label, map_location='cpu')
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers)

# Results directory
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if args.results_dir:
    results_dir = Path(args.results_dir)
else:
    results_dir = Path('results') / f'mnist_extraction_{timestamp}'
results_dir.mkdir(parents=True, exist_ok=True)

# CSV log
log_path = results_dir / f'extraction_log_{timestamp}.csv'
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss', 'test_accuracy', 'agreement_accuracy'])

# Model / training setup
student = CNN()
student = student.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(student.parameters(), lr=args.lr)

# Training loop with tqdm and logging
train_losses = []
test_accuracies = []
agreement_accuracies = []

best_agreement = -1.0
for epoch in range(1, args.epochs + 1):
    student.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = student(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader)
    train_losses.append(epoch_loss)

    # Test accuracy
    student.eval()
    test_acc = calculate_test_accuracy(student, testloader)
    test_accuracies.append(test_acc)

    # Agreement with victim
    agreement_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            s_out = student(images)
            v_out = victim_model(images)
            _, s_pred = torch.max(s_out.data, 1)
            _, v_pred = torch.max(v_out.data, 1)
            agreement_correct += (s_pred == v_pred).sum().item()
            total += images.size(0)
    agreement = 100.0 * agreement_correct / total
    agreement_accuracies.append(agreement)

    # write CSV
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, epoch_loss, test_acc, agreement])

    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.2f}%, Agreement: {agreement:.2f}%")

    # save best by agreement
    if agreement > best_agreement:
        best_agreement = agreement
        torch.save(student.state_dict(), results_dir / 'student_best_state.pt')

# Save final model
torch.save(student.state_dict(), results_dir / 'student_last_state.pt')

# Plotting
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(range(1, args.epochs+1), train_losses, marker='o')
plt.title('Train Loss')
plt.xlabel('Epoch')

plt.subplot(1,3,2)
plt.plot(range(1, args.epochs+1), test_accuracies, marker='o', color='orange')
plt.title('Test Accuracy')
plt.xlabel('Epoch')

plt.subplot(1,3,3)
plt.plot(range(1, args.epochs+1), agreement_accuracies, marker='s', color='green')
plt.title('Agreement with Victim')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig(results_dir / 'training_plots.png', dpi=100)
plt.close()

print(f"Finished. Artifacts saved to {results_dir}")
