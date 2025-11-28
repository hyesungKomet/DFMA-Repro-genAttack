import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
from utils_cifar import *
from torch.utils.data import random_split
import torch.utils.data as Data

def train_epoch(one_model,one_dataloader,one_optimizer):
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for i, data in enumerate(one_dataloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        one_optimizer.zero_grad()
        outputs = one_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        one_optimizer.step()
        running_loss += loss.item()
    print(f'Loss: {running_loss / len(trainloader)}')

def train_steal_epoch(one_model,one_dataloader,one_optimizer):
    criterion = nn.MSELoss()
    running_loss = 0.0
    for i, data in enumerate(one_dataloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        one_optimizer.zero_grad()
        outputs = one_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        one_optimizer.step()
        running_loss += loss.item()
    print(f'Loss: {running_loss / len(trainloader)}')

def calculate_test_accuracy(one_model, one_dataloader):
    one_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in one_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = one_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(accuracy)

def calculate_aggrement_accuracy(one_model,two_model,one_dataloader):
    one_model.eval()
    two_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in one_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = one_model(images)
            ano_outputs=two_model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, ano_predicted = torch.max(ano_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == ano_predicted).sum().item()
    accuracy = 100 * correct / total
    print(accuracy)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

train_size = int(0.7 * len(trainset))
test_size = len(trainset) - train_size
new_train_dataset, new_steal_dataset = random_split(trainset, [train_size, test_size])
trainloader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
steal_loader = DataLoader(new_steal_dataset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

victim_model = get_vggmodel().cuda()
victim_optimizer = optim.SGD(victim_model.parameters(), lr=0.01)

for k in range(20):
    train_epoch(victim_model, trainloader, victim_optimizer)
    calculate_test_accuracy(victim_model, testloader)

steal_data=None
steal_label=None
victim_model.eval()
with torch.no_grad():
    for images, labels in steal_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = victim_model(images)
        if steal_data is not None:
            steal_data=torch.cat((steal_data,images),dim=0)
            steal_label = torch.cat((steal_label, outputs), dim=0)
        else:
            steal_data = images
            steal_label = outputs

steal_dataset = Data.TensorDataset(steal_data, steal_label)
meta_loader = DataLoader(steal_dataset, batch_size=128, shuffle=True)
steal_model = get_vggmodel().cuda()
steal_optimizer = optim.SGD(steal_model.parameters(), lr=0.001)
for k in range(20):
    train_steal_epoch(steal_model, meta_loader, steal_optimizer)
    calculate_test_accuracy(steal_model, testloader)

calculate_aggrement_accuracy(victim_model,steal_model,testloader)