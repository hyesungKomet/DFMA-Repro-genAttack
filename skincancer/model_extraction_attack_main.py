import torch.utils.data
import torchvision.datasets as dset
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as Data
from utils_skin import *


image_size=64
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


val_dataset = dset.ImageFolder(root='./melanoma_cancer_dataset/test', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# Parameters
nz = 100  # Size of generator input (latent vector)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
nc = 3    # Number of channels in the training images (for color images this is 3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



victim_model=torch.load('victim_model.pt')
victim_model.to(device)
victim_model.eval()

generate_img=torch.load('generated_data.pt')
generate_label=torch.load('generated_label.pt')
print(generate_img.shape,generate_label.shape)
torch_dataset = Data.TensorDataset(generate_img, generate_label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True
)

skin_model = get_vggmodel()
criterion = nn.MSELoss()
optimizer = optim.SGD(skin_model.parameters(), lr=1e-5,momentum=0.9)
num_epochs = 100
skin_model.to(device)

for epoch in range(num_epochs):
    skin_model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = skin_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader)}")
    calculate_test_accuracy(skin_model,val_loader)
    calculate_agreement_accuracy(skin_model,victim_model,val_loader)
torch.save(skin_model,'steal_model.pt')
print("Training complete!")