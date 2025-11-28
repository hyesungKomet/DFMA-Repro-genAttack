import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import numpy as np
from self_model import CNN,ComplexAutoencoder,FullyConnectedNN,MetaAttack
import torch.utils.data as Data
import torchvision
from torch.utils.data import random_split
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

torch.manual_seed(123)

def train_shadow(s_model,s_optimizer,train_loader,epochs):
    s_model.train()
    for epoch in range(epochs):  # 迭代8次
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            s_optimizer.zero_grad()
            outputs = s_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            s_optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

def calculate_test_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(accuracy)

def calculate_roc_auc(model, data_loader, device='cuda:0'):
    model.eval()  # 设置模型为评估模式
    true_labels = []
    predictions = []
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 对于sigmoid输出，获取正类的概率
            probs = outputs.squeeze().cpu().numpy()
            predictions.extend(probs)
            labels = labels.cpu().numpy()
            true_labels.extend(labels)
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)


def calculate_f1(model, data_loader, device='cuda:0', threshold=0.5, average='binary'):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > threshold).float()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    f1 = f1_score(true_labels, predictions, average=average)
    print(f'F1 Score: {f1:.4f}')

def calculate_btest_accuracy(model, data_loader, device='cuda:0'):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 二分类任务模型输出处理
            # 如果模型是输出单个概率值（通过sigmoid），需要根据阈值（通常是0.5）判断类别
            predicted = (outputs > 0.5).float()  # 将概率>0.5的视为类别1，否则为0
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

def calculate_tpr_fpr(model, my_loader, device='cuda:0'):
    model.eval()  # 设置模型为评估模式
    correct_result=[]
    result=[]
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for inputs, labels in my_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            for i in range(len(outputs)):
                result.append(outputs[i].item())
                correct_result.append(labels[i].item())
    result=np.array(result)
    correct_result=np.array(correct_result)
    return correct_result,result


def get_meta_data(onemoel,oneloader,label):
    member_data = None
    onemoel=onemoel.cuda()
    onemoel.eval()
    with torch.no_grad():
        for i, data in enumerate(oneloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            outputs = onemoel(inputs)
            if labels.dim() > 1:
                _, labels = torch.max(labels.data, 1)
            for j in range(len(outputs)):
                temp=torch.cat((outputs[j], torch.Tensor([labels[j]]).cuda()), dim=0).unsqueeze(1)
                if member_data is not None:
                    member_data = torch.cat((member_data, temp),dim=1)
                else:
                    member_data=temp
    print(member_data.shape)
    member_data = member_data.transpose(0, 1)
    member_label = [label] * len(member_data)
    member_label = torch.FloatTensor(np.array(member_label))
    return member_data,member_label

def find_kth_largest(arr, k):
    arr_sorted = sorted(arr, reverse=True)
    return arr_sorted[k-1]

batch_size = 100
num_classes = 10
workers=0
ngpu=1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
dataset = trainset
print(f'Total Size of Dataset: {len(dataset)}')

dataloader = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)
test_dataloader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=False
)
# load the victim model
one_model = CNN()
one_model.load_state_dict(torch.load('victim_mnist_model.pth'))
one_model.eval()

new_data=torch.load('generated_data.pt')
new_label=torch.load('generated_label.pt')

torch_dataset = Data.TensorDataset(new_data, new_label)
train_size = int(0.6 * len(torch_dataset))
test_size = len(torch_dataset) - train_size

new_train_dataset, new_test_dataset = random_split(torch_dataset, [train_size, test_size])
# Create DataLoader
new_train_loader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
new_test_loader = DataLoader(new_test_dataset, batch_size=64, shuffle=True)

# Create two shadow models
shadow1=CNN().cuda()
shadow2=CNN().cuda()
criterion = nn.MSELoss()
optimizer1 = optim.SGD(shadow1.parameters(), lr=0.01)
optimizer2 = optim.SGD(shadow2.parameters(), lr=0.02)
# Train shadow models
train_shadow(shadow1,optimizer1,new_train_loader,15)
train_shadow(shadow2,optimizer2,new_train_loader,15)
calculate_test_accuracy(shadow1,new_test_loader)
calculate_test_accuracy(shadow1,new_test_loader)

# Collect training meta-dataset
meta_data,meta_label=None,None
meta_data,meta_label=get_meta_data(shadow1,new_train_loader,1)
meta_data=torch.cat((meta_data,get_meta_data(shadow1,new_test_loader,0)[0]),dim=0)
meta_label=torch.cat((meta_label,get_meta_data(shadow1,new_test_loader,0)[1]),dim=0)

meta_data=torch.cat((meta_data,get_meta_data(shadow2,new_train_loader,1)[0]),dim=0)
meta_label=torch.cat((meta_label,get_meta_data(shadow2,new_train_loader,1)[1]),dim=0)
meta_data=torch.cat((meta_data,get_meta_data(shadow2,new_test_loader,0)[0]),dim=0)
meta_label=torch.cat((meta_label,get_meta_data(shadow2,new_test_loader,0)[1]),dim=0)
print(meta_label.shape,meta_data.shape)

# Collect testing meta-dataset
meta_test_data,meta_test_label=None,None
meta_test_data,meta_test_label=get_meta_data(one_model,dataloader,1)
meta_test_data=torch.cat((meta_test_data[:30000],get_meta_data(one_model,test_dataloader,0)[0]),dim=0)
meta_test_label=torch.cat((meta_test_label[:30000],get_meta_data(one_model,test_dataloader,0)[1]),dim=0)
meta_test_dataset = Data.TensorDataset(meta_test_data, meta_test_label)
meta_test_loader = DataLoader(meta_test_dataset, batch_size=128, shuffle=True)

meta_model=MetaAttack().cuda()
my_cc=nn.BCELoss()
meta_dataset = Data.TensorDataset(meta_data, meta_label)
meta_loader = DataLoader(meta_dataset, batch_size=128, shuffle=True)
meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.1)
for epoch in range(250):
    running_loss = 0.0
    meta_model.train()
    for i, data in enumerate(meta_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        meta_optimizer.zero_grad()
        outputs = meta_model(inputs)
        loss = my_cc(outputs[:,0], labels)
        loss.backward()
        meta_optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(meta_loader)}')
    if epoch%10==0:
        calculate_btest_accuracy(meta_model, meta_test_loader)
        calculate_roc_auc(meta_model, meta_test_loader)
        calculate_f1(meta_model, meta_test_loader)
        c_res, res = calculate_tpr_fpr(meta_model, meta_test_loader)
        zero_indices = np.where(c_res == 0)[0]
        one_indices = np.where(c_res == 1)[0]
        zero_result = res[zero_indices]
        one_result = res[one_indices]
        thr = find_kth_largest(zero_result, int(len(zero_result) * 0.01))
        last_indices = np.where(one_result > thr)[0]
        print('TPR@1%FPR', thr, len(last_indices) / len(one_result))

torch.save(meta_model,'meta_attacker.pt')




