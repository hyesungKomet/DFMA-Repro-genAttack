import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from self_model import MetaAttack
import torch.utils.data as Data
import torchvision
from torch.utils.data import random_split
import torch.optim as optim
from sklearn.metrics import f1_score, roc_curve, auc
from torchvision.models import vgg16


torch.manual_seed(123)
np.random.seed(123)

# ==================== Configuration ====================
batch_size = 100
num_classes = 10
workers = 0
ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

print(f"Using device: {device}")

# ==================== Helper Functions ====================

def train_model(model, optimizer, criterion, train_loader, epochs, model_name="Model"):
    """Train a model (target or shadow)"""
    model.train()
    print(f"\n=== Training {model_name} ===")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'{model_name} - Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model


def evaluate_model(model, data_loader, model_name="Model"):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'{model_name} Accuracy: {accuracy:.2f}%')
    return accuracy


def extract_features(model, data_loader, membership_label):
    """
    Extract features for meta-classifier training.
    Returns: (confidence_vectors, true_labels, membership_labels)
    
    According to paper Section 3.3:
    - Input to attack model A: (T̂(x), y) where T̂(x) is confidence vector
    - Label: 'in' (1) for training set members, 'out' (0) for non-members
    """
    model.eval()
    all_confidences = []
    all_true_labels = []
    all_membership_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Apply softmax to get confidence vectors (probabilities)
            confidences = torch.softmax(outputs, dim=1)
            
            # Store results
            all_confidences.append(confidences.cpu())
            all_true_labels.append(labels)
            all_membership_labels.extend([membership_label] * len(labels))
    
    # Concatenate all batches
    all_confidences = torch.cat(all_confidences, dim=0)
    all_true_labels = torch.cat(all_true_labels, dim=0)
    all_membership_labels = torch.FloatTensor(all_membership_labels)
    
    # Concatenate confidence vector with true label: [confidence(10), label(1)]
    # This creates the input format (T̂(x), y) as described in the paper
    true_labels_expanded = all_true_labels.unsqueeze(1).float()
    features = torch.cat([all_confidences, true_labels_expanded], dim=1)
    
    return features, all_membership_labels


def calculate_metrics(model, data_loader):
    """Calculate various metrics for membership inference"""
    model.eval()
    true_labels = []
    predictions = []
    pred_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()
            preds = (outputs > 0.5).float().squeeze().cpu().numpy()
            
            pred_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            predictions.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
            true_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    pred_probs = np.array(pred_probs)
    
    # Calculate metrics
    accuracy = 100 * np.mean(predictions == true_labels)
    f1 = f1_score(true_labels, predictions, average='binary')
    
    # ROC AUC
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    
    # TPR@1%FPR
    target_fpr = 0.01
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) > 0:
        tpr_at_low_fpr = tpr[idx[-1]]
    else:
        tpr_at_low_fpr = 0.0
    
    return accuracy, f1, roc_auc, tpr_at_low_fpr


# ==================== Data Loading ====================

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load CIFAR-10
print("\n=== Loading CIFAR-10 Dataset ===")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print(f'CIFAR-10 Training set size: {len(trainset)}')
print(f'CIFAR-10 Test set size: {len(testset)}')

# Create data loaders for victim model evaluation
victim_train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
victim_test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)

# ==================== Load Victim Model ====================

print("\n=== Loading Victim Model ===")
victim_model = vgg16(num_classes=10)
victim_model.load_state_dict(torch.load("victim_vgg16_cifar10.pth"))
victim_model = victim_model.to(device)
victim_model.eval()
print("Victim model loaded successfully")

# ==================== Load Synthetic Data ====================

print("\n=== Loading Synthetic Data ===")
synthetic_data = torch.load('generated_data.pt')
synthetic_labels = torch.load('generated_label.pt')
print(f'Synthetic dataset size: {len(synthetic_data)}')

# Split synthetic data: 60% for shadow training, 40% for shadow testing
torch_dataset = Data.TensorDataset(synthetic_data, synthetic_labels)
shadow_train_size = int(0.6 * len(torch_dataset))
shadow_test_size = len(torch_dataset) - shadow_train_size

shadow_train_dataset, shadow_test_dataset = random_split(
    torch_dataset, [shadow_train_size, shadow_test_size]
)

# Create data loaders
shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=64, shuffle=True)
shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=64, shuffle=False)

print(f'Shadow training set size: {shadow_train_size}')
print(f'Shadow test set size: {shadow_test_size}')

# ==================== Train Shadow Models ====================
# Paper Section 3.3: Train shadow models T̂ to mimic target model T

print("\n" + "="*60)
print("STEP 1: Train Shadow Models")
print("="*60)

# Create two shadow models (as in original code)
shadow1 = vgg16(num_classes=10).to(device)
shadow2 = vgg16(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(shadow1.parameters(), lr=0.01, momentum=0.9)
optimizer2 = optim.SGD(shadow2.parameters(), lr=0.01, momentum=0.9)

# Train shadow models
shadow1 = train_model(shadow1, optimizer1, criterion, shadow_train_loader, epochs=15, model_name="Shadow Model 1")
shadow2 = train_model(shadow2, optimizer2, criterion, shadow_train_loader, epochs=15, model_name="Shadow Model 2")

# Evaluate shadow models
print("\n=== Evaluating Shadow Models ===")
evaluate_model(shadow1, shadow_test_loader, "Shadow Model 1")
evaluate_model(shadow2, shadow_test_loader, "Shadow Model 2")

# ==================== Create Meta-Dataset ====================
# Paper Section 3.3: Create training samples ((T̂(x), y), membership_label)

print("\n" + "="*60)
print("STEP 2: Create Meta-Dataset for Attack Model")
print("="*60)

print("\nExtracting features from Shadow Model 1...")
# Shadow Model 1: Training data (members = 1)
features_s1_train, labels_s1_train = extract_features(shadow1, shadow_train_loader, membership_label=1)
print(f"  Training members: {len(features_s1_train)}")

# Shadow Model 1: Test data (non-members = 0)
features_s1_test, labels_s1_test = extract_features(shadow1, shadow_test_loader, membership_label=0)
print(f"  Test non-members: {len(features_s1_test)}")

print("\nExtracting features from Shadow Model 2...")
# Shadow Model 2: Training data (members = 1)
features_s2_train, labels_s2_train = extract_features(shadow2, shadow_train_loader, membership_label=1)
print(f"  Training members: {len(features_s2_train)}")

# Shadow Model 2: Test data (non-members = 0)
features_s2_test, labels_s2_test = extract_features(shadow2, shadow_test_loader, membership_label=0)
print(f"  Test non-members: {len(features_s2_test)}")

# Combine all features for meta-training
meta_features = torch.cat([
    features_s1_train, features_s1_test,
    features_s2_train, features_s2_test
], dim=0)

meta_labels = torch.cat([
    labels_s1_train, labels_s1_test,
    labels_s2_train, labels_s2_test
], dim=0)

print(f"\nTotal meta-dataset size: {len(meta_features)}")
print(f"  Members (label=1): {(meta_labels == 1).sum().item()}")
print(f"  Non-members (label=0): {(meta_labels == 0).sum().item()}")

# Create meta-dataset and loader
meta_dataset = Data.TensorDataset(meta_features, meta_labels)
meta_loader = DataLoader(meta_dataset, batch_size=128, shuffle=True)

# ==================== Train Attack Model ====================
# Paper Section 3.3: Train binary classifier A to predict membership

print("\n" + "="*60)
print("STEP 3: Train Attack Model (Meta-Classifier)")
print("="*60)

# Initialize attack model
attack_model = MetaAttack(input_dim=11).to(device)  # 10 classes + 1 label
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

print("\nTraining attack model...")
num_epochs = 100  # Reduced from 250 for faster training

for epoch in range(num_epochs):
    attack_model.train()
    running_loss = 0.0
    
    for inputs, labels in meta_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = attack_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nNaN detected at epoch {epoch+1}")
            break
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(attack_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(meta_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}')

print("\nAttack model training completed!")

# ==================== Evaluate on Victim Model ====================
# Paper: Apply trained attack model A to victim model T

print("\n" + "="*60)
print("STEP 4: Evaluate Attack on Victim Model")
print("="*60)

# Extract features from victim model
# Use subset of training data (30000 samples) as members
print("\nExtracting features from victim model...")
print("Processing victim training data (members)...")

# Limit to 30000 samples to match original code
victim_train_subset = torch.utils.data.Subset(trainset, range(30000))
victim_train_subset_loader = DataLoader(victim_train_subset, batch_size=batch_size, shuffle=False)

victim_train_features, victim_train_labels = extract_features(
    victim_model, victim_train_subset_loader, membership_label=1
)
print(f"  Members: {len(victim_train_features)}")

print("Processing victim test data (non-members)...")
victim_test_features, victim_test_labels = extract_features(
    victim_model, victim_test_loader, membership_label=0
)
print(f"  Non-members: {len(victim_test_features)}")

# Combine victim features
victim_features = torch.cat([victim_train_features, victim_test_features], dim=0)
victim_labels = torch.cat([victim_train_labels, victim_test_labels], dim=0)

print(f"\nTotal evaluation dataset size: {len(victim_features)}")
print(f"  Members: {(victim_labels == 1).sum().item()}")
print(f"  Non-members: {(victim_labels == 0).sum().item()}")

# Create evaluation loader
eval_dataset = Data.TensorDataset(victim_features, victim_labels)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

# Calculate metrics
print("\n=== Attack Performance Metrics ===")
accuracy, f1, roc_auc, tpr_at_1fpr = calculate_metrics(attack_model, eval_loader)

print(f'Attack Accuracy: {accuracy:.2f}%')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'TPR@1%FPR: {tpr_at_1fpr:.4f}')

# ==================== Save Model ====================

print("\n=== Saving Attack Model ===")
torch.save(attack_model.state_dict(), 'meta_attacker_weights.pth')
torch.save(attack_model, 'meta_attacker.pt')
print("Attack model saved successfully!")

print("\n" + "="*60)
print("Membership Inference Attack Completed!")
print("="*60)