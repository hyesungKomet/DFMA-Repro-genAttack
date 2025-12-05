import copy
import argparse
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import numpy as np
from utils_cifar import *
from pathlib import Path
import own_encoder
import torch.serialization as serialization
from PIL import Image

# Argument parser
parser = argparse.ArgumentParser(description="Generate data from autoencoder with optional augmentation")
parser.add_argument('--model', type=str, default='vgg16', choices=['vgg16', 'resnet18'], 
                    help='Victim model to use (default: vgg16)')
parser.add_argument('--augment', action='store_true', help='Apply augmentation (get_near_sample)')
parser.add_argument('--gen-model', type=str, default='SDXL', help='Name of generative model used (e.g., SDXL, SD1.5)')
parser.add_argument('--output-dir', type=str, default=None, help='Optional output directory to save generated stealing set')
parser.add_argument('--save-images', action='store_true', help='Also save generated samples as image files (PNG)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")
print(f"Model: {args.model}, Augmentation: {args.augment}")
print(f"Generative model: {args.gen_model}")

# Load autoencoder robustly: support full-model checkpoints and state_dict-only files.
auto_path = 'autoencoder.pt'
try:
    model = torch.load(auto_path, map_location=device)
    # if loaded object is a state_dict, model will be a dict
    if isinstance(model, dict):
        ae = own_encoder.AutoEncoder()
        ae.load_state_dict(model)
        model = ae
except Exception as e:
    # Try safe globals context to allow unpickling AutoEncoder if needed
    try:
        with serialization.safe_globals([own_encoder.AutoEncoder]):
            model = torch.load(auto_path, map_location=device, weights_only=False)
    except Exception:
        # Fallback: try loading as weights-only state dict
        state = torch.load(auto_path, map_location=device, weights_only=True)
        ae = own_encoder.AutoEncoder()
        ae.load_state_dict(state)
        model = ae

model = model.to(device)
model.eval()
# Load victim model based on argument. Support both full-model file and state_dict.
if args.model == 'vgg16':
    victim_path = './results/vgg19/vgg19_victim_cifar10_state.pt'
    model_builder = get_vggmodel
elif args.model == 'resnet18':
    victim_path = './results/resnet18/resnet18_victim_cifar10_state.pt'
    model_builder = get_resnet18model
else:
    raise ValueError(f"Unknown model: {args.model}")

victim_obj = torch.load(victim_path, map_location=device)
if isinstance(victim_obj, dict):
    victim_model = model_builder()
    victim_model.load_state_dict(victim_obj)
else:
    victim_model = victim_obj

victim_model = victim_model.to(device)
victim_model.eval()

data_transform = transforms.Compose([transforms.Normalize((0,0,0),(1,1,1))])
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                transforms.Normalize((0,0,0),(1,1,1))])
train_dataset = datasets.ImageFolder(root=f'./generated_data_{args.gen_model}', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train_img = None
train_lab = None
for idx, (data, target) in enumerate(train_loader):
    if train_img is not None:
        train_img = torch.cat([train_img, data], dim=0)
        train_lab = torch.cat([train_lab, target], dim=0)
    else:
        train_img = data
        train_lab = target

datas, labels = [], []
cnt = 0
for i in range(len(train_img)):
    data, label = train_img[i].reshape(1, 3, 32, 32), train_lab[i]
    data, label = data.to(device), label.to(device)
    with torch.no_grad():
        pre = victim_model(data)
        output = torch.argmax(pre)
        if output == label:
            datas.append(data)
            labels.append(label)
print(f"Collected {len(datas)} correct samples")

def get_near_sample(img, target):
    tep = copy.deepcopy(img)
    tep = tep.reshape(1, 3, 32, 32)
    tep = tep.to(device)
    target = target.to(device)
    with torch.no_grad():
        pre = victim_model(tep)
        nears_lab = pre
        nears = tep
        for i in range(20):
            rand_noise = np.random.uniform(0, 0.02, (1, 64, 8, 8))
            rand_noise = rand_noise * (i + 1)
            rand_noise = torch.Tensor(rand_noise).to(device)
            feat = model.get_feature(tep)
            feat = feat + rand_noise
            reverse_img = model.reverse_img(feat)
            reverse_img = data_transform(reverse_img)
            prediction = victim_model(reverse_img)
            if target == torch.argmax(prediction):
                nears = torch.cat([nears, reverse_img], dim=0)
                nears_lab = torch.cat([nears_lab, prediction], dim=0)
            else:
                break
    print(f"Generated {nears_lab.shape[0]} samples from 1 image")
    return nears, nears_lab

if args.augment:
    print("Applying augmentation with get_near_sample...")
    new_data, new_label = None, None
    for i in range(len(datas)):
        print(f"Processing sample {i+1}/{len(datas)}")
        tep, tep_lab = get_near_sample(datas[i], labels[i])
        if new_data is not None:
            new_data = torch.cat([new_data, tep], dim=0)
            new_label = torch.cat([new_label, tep_lab], dim=0)
        else:
            new_data = tep
            new_label = tep_lab
else:
    print("No augmentation - using collected samples as is")
    new_data = torch.cat(datas, dim=0)
    new_label = torch.stack(labels, dim=0)

print(f"Final data shape: {new_data.shape}, labels shape: {new_label.shape}")
# save to stealing_set/{gen_model}/{model}_{augment}/generated_data.pt
if args.output_dir:
    out_dir = Path(args.output_dir)
else:
    out_dir = Path('stealing_set') / args.gen_model / f"{args.model}_{args.augment}"
out_dir.mkdir(parents=True, exist_ok=True)

data_path = out_dir / 'generated_data.pt'
label_path = out_dir / 'generated_label.pt'
torch.save(new_data.cpu(), data_path)
torch.save(new_label.cpu(), label_path)
print(f"Saved {data_path} and {label_path}")

# Optionally save images
if args.save_images:
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # new_data is expected in [0,1] (transform uses Normalize((0,0,0),(1,1,1)))
    if new_label.dim() > 1 and new_label.shape[1] > 1:
        label_idx = torch.argmax(new_label, dim=1).cpu().numpy()
    else:
        label_idx = new_label.cpu().numpy().astype(int)

    for i in range(new_data.shape[0]):
        img_t = new_data[i].cpu().numpy()
        img_np = np.clip(np.transpose(img_t, (1, 2, 0)), 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        cls = int(label_idx[i]) if i < len(label_idx) else -1
        cls_dir = images_dir / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)
        img_path = cls_dir / f"img_{i:06d}_cls{cls}.png"
        Image.fromarray(img_uint8).save(img_path)

    print(f"Saved images to {images_dir}")