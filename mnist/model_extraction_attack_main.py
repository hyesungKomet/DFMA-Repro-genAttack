import torch
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
from utils_mnist import *
import numpy as np
import torch.utils.data as Data
import torchvision
import torch.optim as optim

torch.manual_seed(123)

# Hyperparameter
workers = 0
batch_size = 100
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
# load some test samples of mnist
transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)
# load the trained victim model
victim_model = CNN()
victim_model.load_state_dict(torch.load('victim_mnist_model.pth'))
victim_model.eval()
# load the generated dataset
new_data=torch.load('generated_data.pt')
new_label=torch.load('generated_label.pt')
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True
)
# train the stealing model
new_model = CNN()
new_model.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.005)
for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        outputs = new_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(loader)}')
    calculate_test_accuracy(new_model,testloader)
calculate_agreement_accuracy(victim_model,new_model,testloader)
