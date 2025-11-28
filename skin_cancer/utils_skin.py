import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torchvision.models import vgg19
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Decoder(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, 512 * 4 * 4)
        self.main = nn.Sequential(
            # 输入是 (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),  # (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # (ngf) x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),  # (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), 512, 4, 4)
        decoded = self.main(x)
        return decoded

class DeconvModel(nn.Module):
    def __init__(self):
        super(DeconvModel, self).__init__()
        # Initial linear layer to expand the dimensions
        self.initial_layer = nn.Linear(2, 256 * 8 * 8)

        # Deconvolution layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # output size: [128, 16, 16]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # output size: [64, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # output size: [32, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.3)
        )

        # Final layer to get the desired output dimensions: (3, 64, 64)
        self.final_layer = nn.Conv2d(32, 3, kernel_size=1)  # output size: [3, 64, 64]

    def forward(self, x):
        # Reshape and prepare input for deconvolution layers
        x = self.initial_layer(x)
        x = x.view(-1, 256, 8, 8)  # Reshape to [batch_size, channels, height, width]

        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        # Final layer to get the desired output shape
        x = self.final_layer(x)
        return x


class LargeDeconvModel(nn.Module):
    def __init__(self):
        super(LargeDeconvModel, self).__init__()

        # Initial linear layer to expand the dimensions
        self.initial_layer = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)  # From (512, 2, 2) to (256, 2, 2)

        # Deconvolution layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # From (256, 2, 2) -> (128, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # From (128, 4, 4) -> (64, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # From (64, 8, 8) -> (32, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.3)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # From (32, 16, 16) -> (16, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.3)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # From (16, 32, 32) -> (3, 64, 64)
            nn.Tanh()  # Use Tanh to normalize output to [-1, 1], for example, if that's desired.
        )

    def forward(self, x):
        # Apply initial layer to expand the channels
        x = self.initial_layer(x)

        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x

class TwoMetaAttack(nn.Module):
    def __init__(self):
        super(TwoMetaAttack, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def get_vggmodel():
    vgg16 = vgg19(pretrained=True)
    for param in vgg16.parameters():
        param.requires_grad = True
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, 2)
    return vgg16

def get_testacc(model,val_loader,criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {correct / total}")

def calculate_test_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy:',accuracy)

def calculate_agreement_accuracy(one_model,two_model,one_dataloader):
    one_model.cuda()
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
    print('Agreement:',accuracy)

def random_partation(one_data,one_label):
    indices = torch.randperm(one_data.size(0))

    midpoint = int(len(indices) *0.7)
    first_data = one_data[indices[:midpoint]]
    second_data = one_data[indices[midpoint:]]

    first_label = one_label[indices[:midpoint]]
    second_label = one_label[indices[midpoint:]]

    first_dataset = Data.TensorDataset(first_data,first_label)
    second_dataset = Data.TensorDataset(second_data, second_label)

    first_loader = DataLoader(first_dataset, batch_size=64, shuffle=True)
    second_loader = DataLoader(second_dataset, batch_size=64, shuffle=False)
    return first_loader,second_loader

def train_shadow(s_model,s_optimizer,s_criterion,my_loader,epochs):
    s_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(my_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            s_optimizer.zero_grad()
            outputs = s_model(inputs)
            loss = s_criterion(outputs, labels)
            loss.backward()
            s_optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(my_loader)}')

def train_victim(s_model,s_optimizer,s_criterion,my_loader,epochs):
    s_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(my_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            s_optimizer.zero_grad()
            outputs = s_model(inputs)
            loss = s_criterion(outputs, labels)
            loss.backward()
            s_optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(my_loader)}')

def find_kth_largest(arr, k):
    arr_sorted = sorted(arr, reverse=True)
    return arr_sorted[k-1]

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
    zero_indices = np.where(correct_result == 0)[0]
    one_indices = np.where(correct_result == 1)[0]
    zero_result = result[zero_indices]
    one_result = result[one_indices]
    thr = find_kth_largest(zero_result, int(len(zero_result) * 0.05))
    last_indices = np.where(one_result > thr)[0]
    print('TPR:',thr, len(last_indices) / len(one_result))

def get_meta_baseline(onemoel,oneloader,label):
    member_data = None
    onemoel.eval()
    with torch.no_grad():
        for k, data in enumerate(oneloader, 0):
            inputs, labels = data
            inputs,labels = inputs.cuda(),labels.cuda()
            outputs = onemoel(inputs)
            for i in range(len(outputs)):
                temp=torch.cat((outputs[i], torch.Tensor([labels[i]]).cuda()), dim=0).unsqueeze(1)
                if member_data is not None:
                    member_data = torch.cat((member_data, temp),dim=1)
                else:
                    member_data=temp
    member_data=member_data.transpose(0, 1)
    member_label = [label] * len(member_data)
    member_label = torch.FloatTensor(np.array(member_label))
    return member_data,member_label

def get_meta_data(onemoel,oneloader,label):
    member_data = None
    onemoel.eval()
    with torch.no_grad():
        for k, data in enumerate(oneloader, 0):
            inputs, labels = data
            _, idx_labels = torch.max(labels.data, 1)
            inputs,idx_labels = inputs.cuda(),idx_labels.cuda()
            outputs = onemoel(inputs)
            for i in range(len(outputs)):
                #print(outputs[i].shape,torch.Tensor([labels[i]]).shape)
                temp=torch.cat((outputs[i], torch.Tensor([idx_labels[i]]).cuda()), dim=0).unsqueeze(1)
                if member_data is not None:
                    member_data = torch.cat((member_data, temp),dim=1)
                else:
                    member_data=temp
    member_data=member_data.transpose(0, 1)
    member_label = [label] * len(member_data)
    member_label = torch.FloatTensor(np.array(member_label))
    return member_data,member_label

def calculate_test_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(accuracy)

from sklearn.metrics import roc_curve, auc
def calculate_roc_auc(model, my_loader, device='cuda:0'):
    model.eval()  # 设置模型为评估模式
    true_labels = []
    predictions = []
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for inputs, labels in my_loader:
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

from sklearn.metrics import f1_score

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

def calculate_btest_accuracy(model, my_loader, device='cuda:0'):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for inputs, labels in my_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 二分类任务模型输出处理
            # 如果模型是输出单个概率值（通过sigmoid），需要根据阈值（通常是0.5）判断类别
            predicted = (outputs > 0.5).float()  # 将概率>0.5的视为类别1，否则为0
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')