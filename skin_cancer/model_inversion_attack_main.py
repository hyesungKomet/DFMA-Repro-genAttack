import torch.utils.data
import torchvision.datasets as dset
import torch, torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.utils.data as Data
from utils_skin import *

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_output(one_model,test_img):
    with torch.no_grad():
        features_output = one_model.features(test_img)
        # pooled_output = one_model.avgpool(features_output)
        # pooled_output = pooled_output.view(pooled_output.size(0), -1)
        # fc1_output = one_model.classifier[0](pooled_output)
    return features_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
victim_model=torch.load('victim_model.pt')
victim_model.to(device)
victim_model.eval()
#
# old_data=torch.load('generated_data.pt')
# old_dataset=Data.TensorDataset(old_data,old_data)
#
# new_data=None
# new_label=None
# for idx,(image,_) in enumerate(old_dataset):
#     temp = image.reshape(1, 3, 64, 64)
#     temp=temp.to(device)
#     temp_feat = get_output(victim_model, temp)
#     if new_label is not None:
#         new_data=torch.cat((new_data, temp), dim=0)
#         new_label=torch.cat((new_label,temp_feat),dim=0)
#     else:
#         new_data=temp
#         new_label=temp_feat
#
# print(new_data.shape,new_label.shape)
# torch.save(new_data,'reverse_data.pt')
# torch.save(new_label,'reverse_label.pt')

generate_img=torch.load('reverse_data.pt')
generate_label=torch.load('reverse_label.pt')

print(generate_img.shape,generate_label.shape)
torch_dataset = Data.TensorDataset(generate_img, generate_label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True
)

dataiter = iter(loader)
images, idx_labels = next(dataiter)
print("Original Images")
imshow(torchvision.utils.make_grid(images.cpu().detach()))

inversion_model=LargeDeconvModel()
inversion_model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(inversion_model.parameters(), lr=5e-4)
num_epochs = 1500

for epoch in range(num_epochs):
    inversion_model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = inversion_model(labels)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader)}")
    if epoch%500==499:
        inversion_model.eval()
        with torch.no_grad():
            reconstructed = inversion_model(idx_labels.to(device))
            imshow(torchvision.utils.make_grid(reconstructed.cpu().detach()))
            print(criterion(images.cpu().detach(), reconstructed.cpu().detach()))

torch.save(inversion_model,)