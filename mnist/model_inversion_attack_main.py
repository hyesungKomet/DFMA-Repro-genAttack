import torch
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
from utils_mnist import *
import torch.utils.data as Data
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(123)
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Hyperparameter
batch_size = 100
num_classes = 10
workers=0
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
victim_model=victim_model.cuda()
victim_model.eval()


new_data=torch.load('generated_data.pt')
new_label=torch.load('generated_label.pt')
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True
)
# train the reversed model
new_model = Reverse_Generator()
new_model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(new_model.parameters(), lr=0.01)
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        outputs = new_model(labels)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(loader)}')

# testing
demo_imgs=[]
demo_labs=[]
for i, data in enumerate(testset):
    if i%10==0:
        demo_imgs.append(data[0])
        demo_labs.append(data[1])

new_model.eval()
loss_value=0.0
count=0
with torch.no_grad():
    for i in range(len(demo_imgs)):
        test_img=demo_imgs[i]
        test_img=test_img.cuda()
        inter_output=victim_model(test_img)
        reverse_output=new_model(inter_output)
        loss_value+=criterion(reverse_output[0],test_img)
        output=victim_model(reverse_output[0])
        _, output = torch.max(output.data, 1)
        if output==demo_labs[i]:
            count=count+1
        if i%100==0:
            plt.imshow(reverse_output[0][0].cpu())
            plt.show()
print('MSE_Loss:',loss_value/len(demo_imgs))
print('Accuracy:',count/len(demo_imgs))
torch.save(new_model,'inversion.pt')