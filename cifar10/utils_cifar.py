import torch
import torch.nn as nn
from torchvision.models import vgg19
from sklearn.metrics import f1_score
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader

ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else ('mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else 'cpu'))


class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.fc = nn.Linear(4096, 512 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # (3, 32, 32)
            nn.Tanh()  # [-1, 1] 范围的输出
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)  # 调整形状为 (batch_size, 512, 4, 4)
        x = self.deconv(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Fully connected layer to map the latent vector to a feature map
        self.fc = nn.Linear(4096, 64 * 8 * 8)  # 64 channels to start with and target 8x8 spatial dimension

        # Define the deconvolution (transposed convolution) layers
        self.decoder = nn.Sequential(
            # First ConvTranspose2d layer (from 64 to 32 channels)
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1, padding=3),  # (32, 8, 8)
            nn.ReLU(),

            # Second ConvTranspose2d layer (from 32 to 16 channels)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 16, 16)
            nn.ReLU(),

            # Third ConvTranspose2d layer (from 16 to 3 channels)
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (3, 32, 32)
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        # Pass through the fully connected layer to reshape the input
        x = self.fc(x)
        x = x.view(x.size(0), 64, 8, 8)  # Reshape to (batch_size, 64, 8, 8)

        # Pass through the deconvolution layers to get the output image
        x = self.decoder(x)
        return x

class ComplexAutoencoder(nn.Module):
    def __init__(self):
        super(ComplexAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [batch, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [batch, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # [batch, 64, 2, 2]
        )

        # Bottleneck
        self.fc1 = nn.Linear(64 * 2 * 2, 10)  # [batch, 10]
        self.fc2 = nn.Linear(10, 64 * 2 * 2)  # [batch, 256]

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数以输出在 [0, 1] 范围的像素值
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getFeature(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def getReverse(self,x):
        x = self.fc2(x)
        x = x.view(x.size(0), 64, 2, 2)
        x = self.decoder(x)
        return x


class RealDecoder(nn.Module):
    def __init__(self):
        super(RealDecoder, self).__init__()

        # 反卷积层
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                                          padding=1)  # (batch_size, 512, 1, 1) -> (batch_size, 256, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                                          padding=1)  # (batch_size, 256, 2, 2) -> (batch_size, 128, 4, 4)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                                          padding=1)  # (batch_size, 128, 4, 4) -> (batch_size, 64, 8, 8)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2,
                                          padding=1)  # (batch_size, 64, 8, 8) -> (batch_size, 3, 16, 16)
        self.deconv5 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2,
                                          padding=1)  # (batch_size, 3, 16, 16) -> (batch_size, 3, 32, 32)
        self.batch=nn.Sigmoid()

    def forward(self, x):
        # 直接使用反卷积层逐步恢复图像空间分辨率
        x = self.deconv1(x)  # (batch_size, 256, 2, 2)
        x = self.deconv2(x)  # (batch_size, 128, 4, 4)
        x = self.deconv3(x)  # (batch_size, 64, 8, 8)
        x = self.deconv4(x)  # (batch_size, 3, 16, 16)
        x = self.deconv5(x)  # (batch_size, 3, 32, 32)
        x=self.batch(x)
        return x

class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()

        self.fc1 = nn.Linear(4096, 1024)  # 将512维特征转为512维
        self.fc2 = nn.Linear(1024, 3 * 32 * 32)  # 将512维特征转为3 * 32 * 32
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # (batch_size, 512)
        x = self.fc2(x)  # (batch_size, 3 * 32 * 32)
        x = x.view(-1, 3, 32, 32)
        x = self.sigmoid(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=32,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=32,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=3,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=3,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),  # activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_feature(self,x):
        encoded = self.encoder(x)
        return encoded

    def reverse_img(self,x):
        decoded = self.decoder(x)
        return decoded

def test_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_vggmodel():
    vgg16 = vgg19(pretrained=True)
    for param in vgg16.parameters():
        param.requires_grad = True
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, 10)
    return vgg16


def get_resnet18model():
    from torchvision.models import resnet18
    resnet = resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = True
    # Replace final FC layer for 10 classes (CIFAR-10)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 10)
    return resnet


def calculate_agreement_accuracy(one_model,two_model,one_dataloader):
    one_model.to(device)
    one_model.eval()
    two_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in one_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = one_model(images)
            ano_outputs=two_model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, ano_predicted = torch.max(ano_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == ano_predicted).sum().item()
    accuracy = 100 * correct / total
    print('Agreement:',accuracy)

def calculate_baseline(model, data_loader):
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

def train_baseline(s_model,s_optimizer,my_loader,epochs):
    s_model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(my_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            s_optimizer.zero_grad()
            outputs = s_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            s_optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(my_loader)}')

def calculate_test_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            _, idx_labels = torch.max(labels.data, 1)
            images, idx_labels = images.to(device), idx_labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += idx_labels.size(0)
            correct += (predicted == idx_labels).sum().item()
    accuracy = 100 * correct / total
    print(accuracy)

def train_shadow(s_model,s_optimizer,my_loader,epochs):
    s_model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(my_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            s_optimizer.zero_grad()
            outputs = s_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            s_optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(my_loader)}')


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


def calculate_btest_metrics(model, my_loader, device='cuda:0'):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for inputs, labels in my_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # 将概率>0.5的视为类别1，否则为0
            predicted = predicted.view(-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            TP += int(((predicted == 1) & (labels == 1)).sum().item())
            FP += int(((predicted == 1) & (labels == 0)).sum().item())
            TN += int(((predicted == 0) & (labels == 0)).sum().item())
            FN += int(((predicted == 0) & (labels == 1)).sum().item())

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    print(f'True Positive Rate (TPR): {TPR:.2f}')
    print(f'False Positive Rate (FPR): {FPR:.2f}')


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

def get_baseline_output(mymodel,mydata):
    new_new_label=None
    mymodel.eval()
    with torch.no_grad():
        for i in range(len(mydata)):
            output=mymodel(mydata[i].unsqueeze(0))
            if new_new_label is not None:
                new_new_label=torch.cat((new_new_label,output),dim=0)
            else:
                new_new_label=output
    return new_new_label

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

def get_meta_baseline(onemoel,oneloader,label):
    member_data = None
    onemoel.eval()
    with torch.no_grad():
        for k, data in enumerate(oneloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = onemoel(inputs)
            for i in range(len(outputs)):
                temp = torch.cat((outputs[i], torch.tensor([labels[i]], device=device)), dim=0).unsqueeze(1)
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
            inputs, idx_labels = inputs.to(device), idx_labels.to(device)
            outputs = onemoel(inputs)
            for i in range(len(outputs)):
                temp = torch.cat((outputs[i], torch.tensor([idx_labels[i]], device=device)), dim=0).unsqueeze(1)
                if member_data is not None:
                    member_data = torch.cat((member_data, temp),dim=1)
                else:
                    member_data=temp
    member_data=member_data.transpose(0, 1)
    member_label = [label] * len(member_data)
    member_label = torch.FloatTensor(np.array(member_label))
    return member_data,member_label