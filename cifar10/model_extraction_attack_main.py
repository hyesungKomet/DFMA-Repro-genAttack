import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as Data
from utils_cifar import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_vgg = get_vggmodel().to(device)

transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0,0,0),(1,1,1))])


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

victim_model=torch.load('victim_model.pt')
victim_model.eval()

new_data=torch.load('generated_data.pt')
new_label=torch.load('generated_label.pt')
print(new_data.shape,new_label.shape)
new_dataset=Data.TensorDataset(new_data,new_label)
new_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(my_vgg.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):
    running_loss = 0.0
    my_vgg.train()
    for i, data in enumerate(new_loader, 0):
        inputs, labels = data
        # inputs=train_transform(inputs)
        inputs, labels=inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = my_vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(new_loader)}")
    accuracy = test_accuracy(my_vgg, testloader)
    print(f"Accuracy of the network on the test images: {accuracy}%")
    calculate_agreement_accuracy(my_vgg,victim_model,testloader)
