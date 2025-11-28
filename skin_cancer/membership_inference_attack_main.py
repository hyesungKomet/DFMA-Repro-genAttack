from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from utils_skin import *

ngpu = 1
batch_size=64
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

full_train_dataset = ImageFolder(root='./melanoma_cancer_dataset/train', transform=transform)

train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
print(len(train_dataset),len(test_dataset))

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

generate_imgs=torch.load('generated_data.pt')
generate_labs=torch.load('generated_label.pt')

victim_model = get_vggmodel().to(device)

shadow1=get_vggmodel().to(device)
shadow2=get_vggmodel().to(device)
optimizer1 = optim.SGD(shadow1.parameters(), lr=1e-5,momentum=0.9)
optimizer2 = optim.SGD(shadow2.parameters(), lr=2e-5,momentum=0.9)
my_criterion=nn.MSELoss()

new_data=torch.load('generated_data.pt')
new_label=torch.load('generated_label.pt')

train_one_loader,test_one_loader=random_partation(new_data,new_label)
train_two_loader,test_two_loader=random_partation(new_data,new_label)

train_shadow(shadow1,optimizer1,my_criterion,train_one_loader,100)
train_shadow(shadow2,optimizer2,my_criterion,train_two_loader,100)


victim_model = get_vggmodel().to(device)
victim_criterion = nn.CrossEntropyLoss()
victim_optimizer = optim.SGD(victim_model.parameters(), lr=0.01)
train_shadow(victim_model,victim_optimizer,victim_criterion,train_dataloader,20)

meta_data,meta_label=None,None
meta_data,meta_label=get_meta_data(shadow1,train_one_loader,1)
meta_data=torch.cat((meta_data,get_meta_data(shadow1,test_one_loader,0)[0]),dim=0)
meta_label=torch.cat((meta_label,get_meta_data(shadow1,test_one_loader,0)[1]),dim=0)
meta_data=torch.cat((meta_data,get_meta_data(shadow2,train_two_loader,1)[0]),dim=0)
meta_label=torch.cat((meta_label,get_meta_data(shadow2,train_two_loader,1)[1]),dim=0)
meta_data=torch.cat((meta_data,get_meta_data(shadow2,test_two_loader,0)[0]),dim=0)
meta_label=torch.cat((meta_label,get_meta_data(shadow2,test_two_loader,0)[1]),dim=0)
print(meta_label.shape,meta_data.shape)
meta_dataset = Data.TensorDataset(meta_data, meta_label)
meta_loader = DataLoader(meta_dataset, batch_size=128, shuffle=True)

meta_test_data,meta_test_label=None,None
meta_test_data,meta_test_label=get_meta_baseline(victim_model,train_dataloader,1)
meta_test_data=torch.cat((meta_test_data,get_meta_baseline(victim_model,test_dataloader,0)[0]),dim=0)
meta_test_label=torch.cat((meta_test_label,get_meta_baseline(victim_model,test_dataloader,0)[1]),dim=0)
print(meta_test_label.shape,meta_test_data.shape)

meta_test_dataset = Data.TensorDataset(meta_test_data, meta_test_label)
meta_test_loader = DataLoader(meta_test_dataset, batch_size=128, shuffle=False)

meta_model=TwoMetaAttack().to(device)
my_cc=nn.BCELoss()
meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.1)
for epoch in range(1500):
    running_loss = 0.0
    meta_model.train()
    for i, data in enumerate(meta_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
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
        calculate_tpr_fpr(meta_model, meta_test_loader)

torch.save(meta_model,'meta_attacker.pt')