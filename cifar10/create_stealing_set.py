#python create_stealing_set.py --victim-arch vgg16 --gen-model SDXL --augment --save-images
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import numpy as np
import torch.nn.functional as F
import own_encoder
from own_encoder import AutoEncoder
import torch.serialization as serialization
from utils_cifar import *
import argparse
from tqdm import tqdm
from pathlib import Path
import datetime
import csv
from PIL import Image

# Device detection (prefer CUDA -> MPS -> CPU)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")
# generate some neighbor samples - augmentation
def get_near_sample(img, target, max_trials=20):
    """Generate neighbor samples by adding small noise in latent space and reversing through the autoencoder.
    Returns concatenated images and corresponding predictions from victim model.
    """
    tep = copy.deepcopy(img).reshape(1, 3, 32, 32).to(device)
    target = target.to(device)
    with torch.no_grad():
        # initial prediction from victim model
        pred0 = vgg16_model(tep)
        nears_lab = pred0
        nears = tep
        for i in range(max_trials):
            rand_noise = np.random.uniform(0, 0.02, (1, 64, 8, 8))
            rand_noise = rand_noise * (i + 1)
            rand_noise = torch.tensor(rand_noise, dtype=torch.float32).to(device)
            feat = ae_model.get_feature(tep)
            feat = feat + rand_noise
            reverse_img = ae_model.reverse_img(feat)
            # apply normalization transform expected by victim model
            reverse_img_t = data_transform(reverse_img)
            prediction = vgg16_model(reverse_img_t)
            pred_label = int(torch.argmax(prediction, dim=1).item())
            target_label = int(target.item()) if hasattr(target, 'item') else int(target)
            if pred_label == target_label:
                nears = torch.cat([nears, reverse_img_t], dim=0)
                nears_lab = torch.cat([nears_lab, prediction], dim=0)
            else:
                break
    return nears, nears_lab

# inter-class filtering(distribution shift mitigation)
def inter_class_filter(
    data: torch.Tensor,
    victim_model: nn.Module,
    sigma_factor: float = 3.0,
    batch_size: int = 512,
):
    """
    inter-class filtering (3-sigma rule) 구현.

    - data: (N, 3, 32, 32) image tensor - already normalized
    - victim_model
    - sigma_factor: 3.0 (3-sigma rule)

    return:
    - filtered_data: image tensor after filtering
    - filtered_label: victim logits (stealing model soft label)
    - keep_mask: bool mask (N,) - what samples are left
    """
    victim_model.eval()
    # device = data.device
    device_model = next(victim_model.parameters()).device
    N = data.size(0)

    logits_list = []
    probs_list = []

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc='Inter-class filtering', unit='batch'):
            end = min(start + batch_size, N)
            batch = data[start:end].to(device_model)     
            out = victim_model(batch)                    # (B, C)
            prob = F.softmax(out, dim=1)                 # (B, C)

            logits_list.append(out.cpu())
            probs_list.append(prob.cpu())

        # victim's output (logits)
        logits = torch.cat(logits_list, dim=0)               # (N, C) on CPU
        probs = torch.cat(probs_list, dim=0)                 # (N, C) on CPU
        hard_labels = torch.argmax(probs, dim=1)             # (N,)

        num_classes = probs.shape[1]

        # each class's centroid(mean vector?)
        centroids = []
        for c in range(num_classes):
            mask_c = (hard_labels == c)
            if mask_c.sum() == 0:
                centroids.append(None)
            else:
                centroids.append(probs[mask_c].mean(dim=0))

        keep_mask = torch.ones(N, dtype=torch.bool)

        # for each class i's centroid, 
        # diff class k != i's distance to samples > 3*sigma = outlier
        for i in range(num_classes):
            cen = centroids[i]
            if cen is None:
                continue

            cen = cen.unsqueeze(0)  # (1, C)
            other_mask = (hard_labels != i)
            if other_mask.sum() == 0:
                continue

            probs_other = probs[other_mask]      # (M, C)
            dists = torch.norm(probs_other - cen, dim=1)  # (M,)

            mu = dists.mean()
            sigma = dists.std(unbiased=False)
            if sigma.item() == 0.0:
                
                continue

            lower = mu - sigma_factor * sigma
            upper = mu + sigma_factor * sigma

            # 3-sigma outliers
            outlier_local = (dists < lower) | (dists > upper)

            # global index mapping
            idx_other = torch.nonzero(other_mask, as_tuple=False).squeeze(1)
            out_idx = idx_other[outlier_local]
            keep_mask[out_idx] = False

        filtered_data = data[keep_mask]
        # label - victim logits(like knowledge distillation)
        filtered_label = logits[keep_mask]

    return filtered_data, filtered_label, keep_mask

# Argument: optional output folder
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, default=None, help='Directory to save generated stealing set')
parser.add_argument('--encoder-path', type=str, default='autoencoder.pt', help='Path to autoencoder file')
parser.add_argument('--victim-arch', type=str, default='vgg16', choices=['vgg16','resnet18'], help='Victim model architecture')
parser.add_argument('--gen-model', type=str, default='SD1.5', help='Name of generative model used (e.g., SDXL, SD1.5)')
parser.add_argument('--augment', action='store_true', help='Whether augmentation/neighbor expansion is used')
parser.add_argument('--save-images', action='store_true', help='Also save generated samples as image files (PNG)')
args = parser.parse_args()

print(f"Victim architecture: {args.victim_arch}")

# load the trained autoencoder (support full-model checkpoints and state_dict-only files)
auto_path = args.encoder_path
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

ae_model = model.to(device)
ae_model.eval()

# Load the trained victim model using a standard results path (like generate_data.py)
# Victim path default: results/{victim_arch}/{victim_arch}_victim_cifar10_state.pt
if args.victim_arch == 'vgg16':
    default_victim_path = Path('results') / 'vgg16_victim' / 'vgg16_victim_cifar10_state.pt'
    victim_builder = get_vggmodel
else:
    default_victim_path = Path('results') / 'resnet18_victim' / 'resnet18_victim_cifar10_state.pt'
    victim_builder = get_resnet18model

# Always use the standard results path based on the chosen architecture
victim_path = default_victim_path
try:
    victim_obj = torch.load(victim_path, map_location=device)
    if isinstance(victim_obj, dict):
        victim_model = victim_builder()
        victim_model.load_state_dict(victim_obj)
    else:
        victim_model = victim_obj
except Exception:
    # fallback: try loading to CPU then move
    victim_model = torch.load(victim_path, map_location='cpu')

victim_model = victim_model.to(device)
victim_model.eval()
# alias used in functions
vgg16_model = victim_model

# data transforms
data_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# Load dataset from generated_data_{gen_model} directory
gen_data_dir = Path(f'./generated_data_{args.gen_model}')
if not gen_data_dir.exists():
    raise FileNotFoundError(f"Generated data directory not found: {gen_data_dir}")
train_dataset = datasets.ImageFolder(root=str(gen_data_dir), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_img = None
train_lab = None
for idx, (data, target) in enumerate(tqdm(train_loader, desc='Loading dataset')):
    if train_img is not None:
        train_img = torch.cat([train_img, data], dim=0)
        train_lab = torch.cat([train_lab, target], dim=0)
    else:
        train_img = data
        train_lab = target
# select available samples
datas, labels = [], []
cnt = 0
for i in tqdm(range(len(train_img)), desc='Selecting samples'):
    data, label = train_img[i].reshape(1, 3, 32, 32), train_lab[i]
    data, label = data.to(device), label.to(device)
    with torch.no_grad():
        pre = vgg16_model(data)
        output = torch.argmax(pre, dim=1)
        # check confidence
        if output.item() == int(label.item()) and torch.max(F.softmax(pre, dim=1)) > 0.7:
            datas.append(data)
            labels.append(label)
# If augmentation flag is set, expand available samples via autoencoder neighbor generation.
# Otherwise, use the collected samples as-is to build the PT files.
if args.augment:
    new_data, new_label = None, None
    for i in tqdm(range(len(datas)), desc='Expanding samples'):
        tep, tep_lab = get_near_sample(datas[i], labels[i])
        if new_data is not None:
            new_data = torch.cat([new_data, tep], dim=0)
            new_label = torch.cat([new_label, tep_lab], dim=0)
        else:
            new_data = tep
            new_label = tep_lab
else:
    # No augmentation: stack collected tensors and labels directly
    if len(datas) == 0:
        raise RuntimeError('No samples collected to save (no matching predictions).')
    new_data = torch.cat(datas, dim=0)
    # labels are scalar tensors -> stack to shape (N,)
    new_label = torch.stack(labels, dim=0)

# inter-class filtering!
# after augmentation, stealing candidate
if new_data is None:
    raise RuntimeError("No samples generated after augmentation. Check selection/augmentation settings.")

# (1) augmentation-only ver
augmented_data = new_data.detach().clone()
augmented_label = new_label.detach().clone()

# print augment-only results
print("\n========== AUGMENTATION ONLY ==========")
print(f"[INFO] Total augmented samples: {augmented_data.shape[0]}")

# each class's cnt
if augmented_label.dim() > 1:  # logits
    aug_cls = torch.argmax(augmented_label, dim=1)
else:  # integer labels
    aug_cls = augmented_label

unique, counts = torch.unique(aug_cls.cpu(), return_counts=True)
for u, c in zip(unique, counts):
    print(f"[AUG] Class {int(u)} : {int(c)} samples")
print("========================================\n")

# (2) inter-class filtering ver
filtered_data, filtered_label, keep_mask = inter_class_filter(
    augmented_data,  # victim이 기대하는 정규화가 이미 적용된 이미지
    vgg16_model,     # 또는 args.victim_arch에 맞는 victim 모델
    sigma_factor=3.0
)

# print after filtering
print("\n========== AFTER INTER-CLASS FILTERING ==========")
print(f"[INFO] Total after filtering: {filtered_data.shape[0]}")
print(f"[INFO] Removed by filtering: {augmented_data.shape[0] - filtered_data.shape[0]}")

# 클래스별 개수 출력
if filtered_label.dim() > 1:  # logits
    filt_cls = torch.argmax(filtered_label, dim=1)
else:
    filt_cls = filtered_label

unique_f, counts_f = torch.unique(filt_cls.cpu(), return_counts=True)
for u, c in zip(unique_f, counts_f):
    print(f"[FILT] Class {int(u)} : {int(c)} samples")
print("===============================================\n")

# final stealing set
new_data = filtered_data
new_label = filtered_label

# prepare results directory: stealing_set/{gen_model}/{model_name}_{augment}
if args.output_dir is not None:
    out_dir = Path(args.output_dir)
else:
    out_dir = Path('stealing_set') / args.gen_model / f"{args.victim_arch}_{args.augment}"
out_dir.mkdir(parents=True, exist_ok=True)

# (A) augmentation only
aug_data_path = out_dir / 'augmented_data.pt'
aug_label_path = out_dir / 'augmented_label.pt'
torch.save(augmented_data.cpu(), aug_data_path)
torch.save(augmented_label.cpu(), aug_label_path)
print(f"[INFO] Saved augmented (pre-filter) tensors to {aug_data_path} and {aug_label_path}")

# (B) inter-class filtering
data_path = out_dir / 'generated_data.pt'   
label_path = out_dir / 'generated_label.pt'
torch.save(new_data.cpu(), data_path)
torch.save(new_label.cpu(), label_path)
print(f"[INFO] Saved filtered (final) tensors to {data_path} and {label_path}")

# Optionally save individual PNG images
if args.save_images:
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # denormalize params (same as transform Normalize)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # determine class indices from labels (support soft labels)
    if new_label.dim() > 1 and new_label.shape[1] > 1:
        label_idx = torch.argmax(new_label, dim=1).cpu().numpy()
    else:
        label_idx = new_label.cpu().numpy().astype(int)

    for i in range(new_data.shape[0]):
        img_t = new_data[i].cpu().numpy()
        # img_t is CHW normalized -> denormalize and to HWC
        for c in range(3):
            img_t[c] = img_t[c] * std[c] + mean[c]
        img_np = np.clip(np.transpose(img_t, (1, 2, 0)), 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        cls = int(label_idx[i]) if i < len(label_idx) else -1
        cls_dir = images_dir / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)
        img_path = cls_dir / f"img_{i:06d}_cls{cls}.png"
        Image.fromarray(img_uint8).save(img_path)

    print(f"Saved images to {images_dir}")

# write a small CSV log
log_path = out_dir / 'creation_log.csv'
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'num_initial_images', 'num_selected', 'num_generated'])
    num_initial = int(train_img.shape[0]) if train_img is not None else 0
    num_selected = len(datas)
    num_generated = int(new_data.shape[0]) if new_data is not None else 0
    writer.writerow([timestamp, num_initial, num_selected, num_generated])

print(f"Saved generated data to {data_path} and labels to {label_path}")
print(f"Log written to {log_path}")