import copy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import numpy as np
from utils_cifar import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=torch.load('autoencoder.pt')
model=model.to(device)
model.eval()

vgg16=torch.load('victim_model.pt')
vgg16 = vgg16.to(device)
vgg16.eval()

data_transform=transforms.Compose([transforms.Normalize((0,0,0),(1,1,1))])
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                transforms.Normalize((0,0,0),(1,1,1))])
train_dataset = datasets.ImageFolder(root='./gene', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train_img=None
train_lab=None
for idx, (data,target) in enumerate(train_loader):
    if train_img is not None:
        train_img=torch.cat([train_img,data],dim=0)
        train_lab=torch.cat([train_lab,target],dim=0)
    else:
        train_img = data
        train_lab = target

datas,labels=[],[]
cnt=0
for i in range(len(train_img)):
    data,label=train_img[i].reshape(1,3,32,32),train_lab[i]
    data, label= data.to(device),label.to(device)
    with torch.no_grad():
        pre=vgg16(data)
        output=torch.argmax(pre)
        if output==label:
            datas.append(data)
            labels.append(label)
print(len(datas),len(labels))

def get_near_sample(img,target):
    tep=copy.deepcopy(img)
    tep=tep.reshape(1,3,32,32)
    tep=tep.to(device)
    target=target.to(device)
    with torch.no_grad():
        pre=vgg16(tep)
        nears_lab=pre
        nears = tep
        for i in range(20):
            rand_noise = np.random.uniform(0, 0.02, (1, 64, 8, 8))
            rand_noise=rand_noise*(i+1)
            rand_noise=torch.Tensor(rand_noise).to(device)
            feat=model.get_feature(tep)
            feat=feat+rand_noise
            reverse_img=model.reverse_img(feat)
            # plt.imshow(reverse_img[0].cpu().permute(1,2,0))
            # plt.show()
            reverse_img=data_transform(reverse_img)
            prediction=vgg16(reverse_img)
            if target==torch.argmax(prediction):
                nears=torch.cat([nears,reverse_img],dim=0)
                nears_lab=torch.cat([nears_lab,prediction],dim=0)
            else:
                break
    print(nears_lab.shape)
    return nears,nears_lab

new_data,new_label=None,None
for i in range(len(datas)):
    tep,tep_lab=get_near_sample(datas[i],labels[i])
    if new_data is not None:
        new_data = torch.cat([new_data, tep], dim=0)
        new_label = torch.cat([new_label, tep_lab], dim=0)
    else:
        new_data = tep
        new_label = tep_lab

print(new_data.shape,new_label.shape)
torch.save(new_data,'generated_data.pt')
torch.save(new_label,'generated_label.pt')