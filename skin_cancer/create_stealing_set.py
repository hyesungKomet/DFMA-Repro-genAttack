import os
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.datasets as dset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_skin import device, get_vggmodel

#  Generator
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


#  inter-class filtering (logits based)
def inter_class_filter_from_logits(logits: torch.Tensor, sigma_factor: float = 3.0) -> torch.Tensor:
    """
    logits: (N, C)
    sigma_factor: 3.0 -> 3-sigma rule
    return: keep_mask (N,) bool tensor
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)          # (N, C)
        hard_labels = torch.argmax(probs, dim=1)  # (N,)
        num_classes = probs.shape[1]
        N = probs.size(0)

        centroids = []
        for c in range(num_classes):
            mask_c = (hard_labels == c)
            if mask_c.sum() == 0:
                centroids.append(None)
            else:
                centroids.append(probs[mask_c].mean(dim=0))

        keep_mask = torch.ones(N, dtype=torch.bool)

        for i in range(num_classes):
            cen = centroids[i]
            if cen is None:
                continue
            cen = cen.unsqueeze(0)  # (1, C)

            other_mask = (hard_labels != i)
            if other_mask.sum() == 0:
                continue

            probs_other = probs[other_mask]                   # (M, C)
            dists = torch.norm(probs_other - cen, dim=1)      # (M,)

            mu = dists.mean()
            sigma = dists.std(unbiased=False)
            if sigma.item() == 0.0:
                continue

            lower = mu - sigma_factor * sigma
            upper = mu + sigma_factor * sigma

            outlier_local = (dists < lower) | (dists > upper)
            idx_other = torch.nonzero(other_mask, as_tuple=False).squeeze(1)
            out_idx = idx_other[outlier_local]
            keep_mask[out_idx] = False

    return keep_mask

# model loaders
def load_victim_model(path: str) -> nn.Module:
    """
    victim model:
    - state_dict(dict) -> call get_vggmodel()
    - or entire model
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = get_vggmodel()
        model.load_state_dict(obj)

    model.to(device)
    model.eval()
    return model



def load_generator(path: str, nz: int, ngf: int, nc: int) -> nn.Module:
    """
    GAN generator:
    - state_dict(dict) -> create Generator() and load
    - or entire model (saved with torch.save(netG, path))
    """

    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        netG = obj
    else:
        netG = Generator(nz, ngf, nc)
        netG.load_state_dict(obj)

    netG.to(device)
    netG.eval()
    return netG

#  read class names (ImageFolder)
def get_class_names(train_root: str):
    try:
        tmp_ds = dset.ImageFolder(
            root=train_root,
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
            ])
        )
        class_names = tmp_ds.classes
        print(f"[INFO] Detected classes from ImageFolder: {class_names}")
        return class_names
    except Exception as e:
        print(f"[WARN] Could not load classes from ImageFolder ({e}). Use default names.")
        return None


#  GAN generate for 1 class + get victim logits
def generate_for_class(
    netG: nn.Module,
    victim_model: nn.Module,
    class_idx: int,
    n_samples: int,
    nz: int,
    batch_size: int
):
    """
    netG: GAN for each class
    victim_model: victim classifier
    class_idx: label (ex: 0=benign, 1=malignant)
    n_samples: number of samples to generate
    nz: latent dimension
    batch_size: batch size for generation
    """
    imgs_list = []
    logits_list = []
    labels_list = []

    total = 0
    pbar = tqdm(total=n_samples, desc=f"Generating class {class_idx}")
    while total < n_samples:
        cur_bs = min(batch_size, n_samples - total)
        z = torch.randn(cur_bs, nz, 1, 1, device=device)
        with torch.no_grad():
            fake = netG(z)                        # (B, 3, 64, 64)
            # out = victim_model(fake)             # (B, C)
            victim_dev = next(victim_model.parameters()).device
            out = victim_model(fake.to(victim_dev))  

        imgs_list.append(fake.cpu())
        logits_list.append(out.cpu())
        labels_list.append(torch.full((cur_bs,), class_idx, dtype=torch.long))

        total += cur_bs
        pbar.update(cur_bs)
    pbar.close()

    imgs = torch.cat(imgs_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    hard_labels = torch.cat(labels_list, dim=0)
    return imgs, logits, hard_labels


#  save filtered images(per class folders)
def save_filtered_images(
    images: torch.Tensor,
    logits: torch.Tensor,
    save_root: str,
    class_names=None
):
    """
    images: (N, 3, 64, 64), [-1,1](tanh output)
    logits: (N, C)
    save_root: ./generated_data_GAN
    class_names: ['benign','malignant'] if not, class_0 ..
    """
    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        hard_labels = torch.argmax(probs, dim=1)  # (N,)

    num_classes = logits.size(1)
    if class_names is None or len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    for c in range(num_classes):
        os.makedirs(os.path.join(save_root, class_names[c]), exist_ok=True)

    print("[INFO] Saving filtered images to", save_root)
    for idx in tqdm(range(images.size(0)), desc="Saving filtered images"):
        img_tensor = images[idx]  # (3, 64, 64)
        c_idx = hard_labels[idx].item()
        c_name = class_names[c_idx]

        # [-1,1] -> [0,1]
        img = img_tensor.clamp(-1, 1)
        img = (img + 1) / 2.0  # [0,1]

        pil_img = to_pil_image(img.cpu())
        out_dir = os.path.join(save_root, c_name)
        filename = os.path.join(out_dir, f"{idx:06d}.png")
        pil_img.save(filename)


#  main
def main():
    parser = argparse.ArgumentParser(description="GAN-based synthetic data generation for skin_cancer with augmentation + inter-class filtering")

    parser.add_argument("--victim-path", type=str, default="./results/vgg16_victim/vgg16_victim_skin_state.pt",
                        help="Path to victim model (.pt)")
    parser.add_argument("--gen-malignant-path", type=str, default="./netG_malignant.pt",
                        help="Path to GAN generator for malignant class")
    parser.add_argument("--gen-benign-path", type=str, default="./netG_benign.pt",
                        help="Path to GAN generator for benign class")

    parser.add_argument("--train-root", type=str, default="./melanoma_cancer_dataset/train",
                        help="ImageFolder root to infer class names")
    parser.add_argument("--stealing-root", type=str, default="./stealing_set_GAN",
                        help="Directory to save .pt files (augmented / filtered)")
    parser.add_argument("--images-root", type=str, default="./generated_data_GAN",
                        help="Directory to save filtered images per class")

    parser.add_argument("--num-per-class", type=int, default=2000,
                        help="# of samples to generate per class")
    parser.add_argument("--gen-batch-size", type=int, default=64,
                        help="Batch size for GAN generation")
    parser.add_argument("--nz", type=int, default=100, help="Latent vector size for GAN")
    parser.add_argument("--ngf", type=int, default=64, help="# of generator feature maps")
    parser.add_argument("--nc", type=int, default=3, help="# of image channels")

    parser.add_argument("--sigma-factor", type=float, default=3.0,
                        help="Sigma factor for inter-class filtering")
    parser.add_argument("--no-filter", action="store_true",
                        help="If set, skip inter-class filtering and only save augmented data")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[INFO] Device: {device}")
    print("[INFO] Args:", args)

    # class names
    class_names = get_class_names(args.train_root)

    # model loading
    print("[INFO] Loading victim model...")
    victim_model = load_victim_model(args.victim_path)
    if device.type == "mps":
        victim_model.to("cpu")
        print("[INFO] Using CPU for victim model to avoid MPS adaptive_avg_pool2d issue.")

    print("[INFO] Loading GAN generators...")
    netG_malignant = load_generator(args.gen_malignant_path, args.nz, args.ngf, args.nc)
    netG_benign = load_generator(args.gen_benign_path, args.nz, args.ngf, args.nc)

    # Augmentation(GAN generation)
    print("[INFO] Start GAN-based augmentation...")
    # class_idx 0,1(benign=0, malignant=1)
    mal_imgs, mal_logits, mal_hard = generate_for_class(
        netG=netG_malignant,
        victim_model=victim_model,
        class_idx=1,
        n_samples=args.num_per_class,
        nz=args.nz,
        batch_size=args.gen_batch_size,
    )
    ben_imgs, ben_logits, ben_hard = generate_for_class(
        netG=netG_benign,
        victim_model=victim_model,
        class_idx=0,
        n_samples=args.num_per_class,
        nz=args.nz,
        batch_size=args.gen_batch_size,
    )

    aug_imgs = torch.cat([mal_imgs, ben_imgs], dim=0)
    aug_logits = torch.cat([mal_logits, ben_logits], dim=0)

    # augmented set logging
    with torch.no_grad():
        probs_aug = F.softmax(aug_logits, dim=1)
        aug_hard = torch.argmax(probs_aug, dim=1)
        num_classes = aug_logits.size(1)

        print(f"[LOG] Augmented total samples: {aug_imgs.size(0)}")
        for c in range(num_classes):
            cnt = (aug_hard == c).sum().item()
            cname = class_names[c] if (class_names is not None and c < len(class_names)) else f"class_{c}"
            print(f"  - Class {c} ({cname}): {cnt} samples in augmented set")

    # save augmented .pt
    stealing_root = Path(args.stealing_root)
    stealing_root.mkdir(parents=True, exist_ok=True)

    aug_data_path = stealing_root / "augmented_data.pt"
    aug_label_path = stealing_root / "augmented_label.pt"

    torch.save(aug_imgs, aug_data_path)
    torch.save(aug_logits, aug_label_path)
    print(f"[SAVE] Augmented images saved to {aug_data_path}")
    print(f"[SAVE] Augmented logits saved to {aug_label_path}")

    # filtering
    if args.no_filter:
        print("[INFO] --no-filter flag is set. Skipping inter-class filtering.")
        filtered_imgs = aug_imgs
        filtered_logits = aug_logits
    else:
        print("[INFO] Applying inter-class filtering...")
        keep_mask = inter_class_filter_from_logits(aug_logits, sigma_factor=args.sigma_factor)
        filtered_imgs = aug_imgs[keep_mask]
        filtered_logits = aug_logits[keep_mask]

        print(f"[LOG] Filtering kept {filtered_imgs.size(0)} / {aug_imgs.size(0)} samples.")

    # filtered set loggin
    with torch.no_grad():
        probs_f = F.softmax(filtered_logits, dim=1)
        hard_f = torch.argmax(probs_f, dim=1)
        num_classes_f = filtered_logits.size(1)

        print(f"[LOG] Filtered total samples: {filtered_imgs.size(0)}")
        for c in range(num_classes_f):
            cnt = (hard_f == c).sum().item()
            cname = class_names[c] if (class_names is not None and c < len(class_names)) else f"class_{c}"
            print(f"  - Class {c} ({cname}): {cnt} samples in filtered set")

    # save filtered .pt
    filt_data_path = stealing_root / "filtered_data.pt"
    filt_label_path = stealing_root / "filtered_label.pt"

    torch.save(filtered_imgs, filt_data_path)
    torch.save(filtered_logits, filt_label_path)
    print(f"[SAVE] Filtered images saved to {filt_data_path}")
    print(f"[SAVE] Filtered logits saved to {filt_label_path}")

    # save filtered images
    save_filtered_images(
        images=filtered_imgs,
        logits=filtered_logits,
        save_root=args.images_root,
        class_names=class_names,
    )

    print("[DONE] Generation + augmentation + filtering + saving completed.")


if __name__ == "__main__":
    main()