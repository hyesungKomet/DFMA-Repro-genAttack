import torch
import torch.nn as nn

class MetaAttack(nn.Module):
    """
    Meta-classifier for membership inference attack.
    Takes concatenated model outputs and true label as input,
    outputs probability of being a training member.
    """
    def __init__(self, input_dim=11, hidden_dims=[256, 128, 64]):
        """
        Args:
            input_dim: Input dimension (num_classes + 1 for CIFAR-10: 10 + 1 = 11)
            hidden_dims: List of hidden layer dimensions
        """
        super(MetaAttack, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 11)
               First 10 dims: model output probabilities/logits
               Last 1 dim: true label
        
        Returns:
            Output tensor of shape (batch_size, 1) with membership probability
        """
        return self.model(x)


# Alternative simpler version if needed
class MetaAttackSimple(nn.Module):
    """
    Simpler meta-classifier for membership inference attack.
    """
    def __init__(self, input_dim=11):
        super(MetaAttackSimple, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x