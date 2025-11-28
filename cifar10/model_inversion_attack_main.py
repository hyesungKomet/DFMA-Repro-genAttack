import copy
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import numpy as np
import torch.nn.functional as F
from utils_cifar import *
import torch.utils.data as Data

def get_output(one_model,test_img):
    with torch.no_grad():
        features_output = one_model.features(test_img)
        pooled_output = one_model.avgpool(features_output)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        fc1_output = one_model.classifier[0](pooled_output)
    return fc1_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

victim_model=torch.load('victim_model.pt')
victim_model = victim_model.to(device)
victim_model.eval()

transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                transforms.Normalize((0,0,0),(1,1,1))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dataset = datasets.ImageFolder(root='./gene', transform=transform)


test_img=None
for idx,(image,_) in enumerate(testset):
    test_img=image
    test_img = test_img.to(device)
    test_img = test_img.reshape(1, 3, 32, 32)
    test_feat=get_output(victim_model,test_img)
    if idx>99:
        break

orignial_img=copy.deepcopy(test_img[0])
orignial_img=orignial_img.permute(1, 2, 0)
plt.imshow(orignial_img.cpu().detach().numpy())
plt.show()

# old_data=torch.load('generated_data.pt')
# old_dataset=Data.TensorDataset(old_data,old_data)
#
# new_data=None
# new_label=None
# for idx,(image,_) in enumerate(old_dataset):
#     if idx%2==0:
#         continue
#     temp = image.reshape(1, 3, 32, 32)
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

new_data=torch.load('reverse_data.pt')
new_label=torch.load('reverse_label.pt')
print(new_data.shape,new_label.shape)
reverse_dataset=Data.TensorDataset(new_data,new_label)
reverse_loader = DataLoader(reverse_dataset, batch_size=64, shuffle=True)

model = ComplexAutoencoder()
# model = Decoder()
model=model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2,momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters())

num_epochs = 200
for epoch in range(num_epochs):
    running_loss=0.0
    model.train()
    for img,lab in reverse_loader:
        lab, img=lab.to(device), img.to(device)
        # outputs = model(lab)
        outputs = model(img)
        # outputs=(outputs+1.0)/2.0
        loss = criterion(outputs, img)*10.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(reverse_loader)}")
    if epoch%10==9:
        model.eval()
        with torch.no_grad():
            inverse_img = model(test_img)
            tp = inverse_img[0].permute(1, 2, 0)
            # tp=(tp+1.0)/2.0
            plt.imshow(tp.cpu().detach().numpy())
            plt.show()
