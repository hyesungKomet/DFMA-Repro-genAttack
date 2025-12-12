import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization as serialization

from utils_mnist import *  # CNN, Generator, ComplexAutoencoder

# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

# 2. Helper: victim prediction
def predict_img(img: torch.Tensor, victim_model: nn.Module) -> torch.Tensor:
    """
    img: (B, 1, 28, 28) or (1, 1, 28, 28)
    return: hard label (B,) or (1,)
    """
    victim_model.eval()
    with torch.no_grad():
        prediction = victim_model(img.to(device))
        output = F.softmax(prediction, dim=1)
        result = torch.argmax(output, dim=1)
    return result.cpu()


# 3. Autoencoder based neighbor generation
def generate_near_sample(autoencoder: nn.Module,
                         img: torch.Tensor,
                         low: float,
                         high: float) -> torch.Tensor:
    """
    autoencoder: ComplexAutoencoder (reverse / reconstruction)
    img: (1, 1, 28, 28) tensor, range: [-1, 1]
    low, high: latent noise range
    """
    autoencoder.eval()
    with torch.no_grad():
        # latent extraction
        feat = autoencoder.reverse(img.to(device))  # (1, 128, 4, 4)
        noise = np.random.uniform(low, high, feat.shape).astype(np.float32)
        noise = torch.from_numpy(noise).to(device)
        feat_noisy = feat + noise
        near = autoencoder.reconstruction(feat_noisy)  # Sigmoid -> [0,1]
        return near.cpu()  # (1, 1, 28, 28)

# 4. GAN based image generation
def generate_img(netG: nn.Module,
                 label_1hots: torch.Tensor,
                 class_idx: int) -> torch.Tensor:
    """
    netG: Generator
    label_1hots: (10, 10, 1, 1) one-hot tensor
    class_idx: 0~9
    return: (1, 1, 32, 32) Tanh ([-1,1])
    """
    seed = np.random.randint(0, 100000)
    torch.manual_seed(seed)
    fixed_noise = torch.randn(1, nz, 1, 1)
    fixed_label = label_1hots[class_idx].unsqueeze(0)  # (1, 10, 1, 1)
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise.to(device), fixed_label.to(device))  # (1, 1, 32, 32) Tanh
    return fake.cpu()


def generate_base_imgs_for_class(netG: nn.Module,
                                 label_1hots: torch.Tensor,
                                 victim_model: nn.Module,
                                 class_idx: int,
                                 num_candidates: int = 300) -> list:
    """
    GAN -  generate num_candidates images for class_idx
    return: list of images that victim predicts as class_idx
    (1, 1, 32, 32) tensor
    """
    fake_imgs = []
    for _ in range(num_candidates):
        img = generate_img(netG, label_1hots, class_idx)  # (1, 1, 32, 32)
        pred = predict_img(img, victim_model)             # (1,)
        if int(pred.item()) == class_idx:
            fake_imgs.append(img)
    return fake_imgs


def generate_neighbors_for_class(autoencoder: nn.Module,
                                 victim_model: nn.Module,
                                 base_imgs: list,
                                 class_idx: int,
                                 max_neighbors_per_base: int = 10,
                                 init_low: float = 0.0,
                                 init_high: float = 0.02) -> list:
    """
    base_imgs: [(1,1,28,28) or (1,1,32,32)] list. 28x28 would be better.
               32x32 -> resize to 28x28 before passing.
    return: fake_samples (각각 (1,1,28,28) 텐서)
    """
    fake_samples = []

    for base in base_imgs:
        # base: (1,1, H, W)
        # victim predicts as class_idx
        # origin img included in samples
        fake_samples.append(base[0].cpu())  # (1, 28, 28) or (1, 32, 32)

        low, high = init_low, init_high
        for _ in range(max_neighbors_per_base):
            near = generate_near_sample(autoencoder, base, low, high)  # (1,1,28,28), [0,1]
            # if victim expects [-1,1] range -> scale
            near_scaled = (near - 0.5) * 2.0  # [-1,1]
            pred = predict_img(near_scaled, victim_model)
            if int(pred.item()) == class_idx:
                fake_samples.append(near_scaled[0])  # (1, 28, 28)
                low += 0.02
                high += 0.02
            else:
                break

    return fake_samples


# 5. victim logits extraction in batches
def get_logits_in_batches(model: nn.Module,
                          data: torch.Tensor,
                          batch_size: int = 128) -> torch.Tensor:
    """
    data: (N, 1, 28, 28)
    return: (N, 10) logits
    """
    model.eval()
    logits_list = []
    N = data.size(0)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = data[start:end].to(device)
            out = model(batch)
            logits_list.append(out.cpu())
    return torch.cat(logits_list, dim=0)


# 6. inter-class filtering
def inter_class_filter(
    data: torch.Tensor,
    victim_model: nn.Module,
    sigma_factor: float = 3.0,
    batch_size: int = 512,
):
    """
    data: (N, 1, H, W) image tensor (victim expects normalized input)
    victim_model: CNN
    sigma_factor: 3-sigma rule factor

    return:
        filtered_data: (M, 1, H, W)
        filtered_label: (M, 10) logits
        keep_mask: (N,) bool
    """
    victim_model.eval()
    device_model = next(victim_model.parameters()).device
    N = data.size(0)

    logits_list = []
    probs_list = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = data[start:end].to(device_model)
            out = victim_model(batch)                 # (B, 10)
            prob = F.softmax(out, dim=1)             # (B, 10)
            logits_list.append(out.cpu())
            probs_list.append(prob.cpu())

    logits = torch.cat(logits_list, dim=0)           # (N, 10)
    probs = torch.cat(probs_list, dim=0)             # (N, 10)
    hard_labels = torch.argmax(probs, dim=1)         # (N,)

    num_classes = probs.shape[1]

    # centroid
    centroids = []
    for c in range(num_classes):
        mask_c = (hard_labels == c)
        if mask_c.sum() == 0:
            centroids.append(None)
        else:
            centroids.append(probs[mask_c].mean(dim=0))  # (10,)

    keep_mask = torch.ones(N, dtype=torch.bool)

    # each class i' centroid, for diff class (k != i)
    # distance w/ centroid > 3*sigma -> outlier, remove
    for i in range(num_classes):
        cen = centroids[i]
        if cen is None:
            continue

        cen = cen.unsqueeze(0)        # (1, 10)
        other_mask = (hard_labels != i)
        if other_mask.sum() == 0:
            continue

        probs_other = probs[other_mask]            # (M, 10)
        dists = torch.norm(probs_other - cen, dim=1)  # (M,)

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

    filtered_data = data[keep_mask]
    filtered_label = logits[keep_mask]

    return filtered_data, filtered_label, keep_mask


# 7. main: Argument + whole pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator-path', type=str, default='last_model.pt')
    parser.add_argument('--victim-path', type=str, default='results/cnn/cnn_victim_mnist_state.pth')
    parser.add_argument('--autoencoder-path', type=str, default='autoencoder.pt',
                        help='ComplexAutoencoder state_dict path')
    parser.add_argument('--output-dir', type=str, default='stealing_set_GAN')
    parser.add_argument('--num-candidates', type=int, default=150,
                        help='each class, GAN generated candidate images number')
    parser.add_argument('--target-per-class', type=int, default=140,
                        help='each class, target number of final samples (including base + neighbors). ')
    parser.add_argument('--apply-filter', action='store_true',
                        help='3-sigma inter-class filtering')
    parser.add_argument('--filter-sigma', type=float, default=3.0)
    parser.add_argument('--batch-size-logit', type=int, default=128,
                        help='victim logits extraction batch size')
    args = parser.parse_args()

    # model loading

    # Generator
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(args.generator_path, map_location=device), strict=False)
    netG.eval()

    # victim CNN
    victim_model = CNN().to(device)
    victim_model.load_state_dict(torch.load(args.victim_path, map_location=device))
    victim_model.eval()

    # ComplexAutoencoder
    auto_path = args.autoencoder_path
    try:
        autoencoder = torch.load(auto_path, map_location=device)
        if isinstance(autoencoder, dict):
            autoencoder = ComplexAutoencoder()
            autoencoder.load_state_dict(autoencoder)
    except Exception:
        try:
            with serialization.safe_globals([ComplexAutoencoder]):
                autoencoder = torch.load(auto_path, map_location=device, weights_only=False)
        except Exception:
            state = torch.load(auto_path, map_location=device, weights_only=True)
            autoencoder = ComplexAutoencoder()
            autoencoder.load_state_dict(state)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()


    # label one-hot for G
    label_1hots = torch.zeros(10, 10)
    for i in range(10):
        label_1hots[i, i] = 1
    label_1hots = label_1hots.view(10, 10, 1, 1)

    # sample generation
    all_data = []
    all_cls_labels = []

    for cls in range(10):
        print(f"\n[CLASS {cls}] Generating base images...")
        base_imgs = generate_base_imgs_for_class(
            netG, label_1hots, victim_model,
            class_idx=cls,
            num_candidates=args.num_candidates
        )
        print(f"[CLASS {cls}] Base images kept (predict==cls): {len(base_imgs)}")

        if len(base_imgs) == 0:
            print(f"[WARN] No base images for class {cls}. Skipping...")
            continue

        print(f"[CLASS {cls}] Generating neighbors...")
        near_samples = generate_neighbors_for_class(
            autoencoder, victim_model, base_imgs, class_idx=cls
        )
        print(f"[CLASS {cls}] Total near+base samples before clipping: {len(near_samples)}")

        # target_per_class(if over clip)
        if args.target_per_class is not None and len(near_samples) > args.target_per_class:
            idx = np.random.choice(len(near_samples),
                                   args.target_per_class,
                                   replace=False)
            near_samples = [near_samples[i] for i in idx]

        print(f"[CLASS {cls}] Final used samples: {len(near_samples)}")

        all_data.extend(near_samples)
        all_cls_labels.extend([cls] * len(near_samples))

    if len(all_data) == 0:
        raise RuntimeError("No samples generated for any class.")

    # change to (N, 1, H, W) tensor
    data_tensor = torch.stack(all_data, dim=0)  # (N, 1, H, W)
    cls_tensor = torch.LongTensor(all_cls_labels)  # (N,)

    print("\n[GLOBAL] Total augmented samples:", data_tensor.shape[0])

    # ----- victim logits calculation (augmentation-only label) -----
    logits_all = get_logits_in_batches(
        victim_model, data_tensor,
        batch_size=args.batch_size_logit
    )  # (N, 10)

    # augmentation-only label distribution
    print("\n========== AUGMENTATION ONLY ==========")
    unique, counts = torch.unique(torch.argmax(logits_all, dim=1), return_counts=True)
    for u, c in zip(unique, counts):
        print(f"[AUG] Class {int(u)} : {int(c)} samples")
    print("=======================================\n")

    # ----- inter-class filtering -----
    if args.apply_filter:
        filtered_data, filtered_label, keep_mask = inter_class_filter(
            data_tensor, victim_model,
            sigma_factor=args.filter_sigma
        )
        print("\n========== AFTER INTER-CLASS FILTERING ==========")
        print(f"[INFO] Total after filtering: {filtered_data.shape[0]}")
        print(f"[INFO] Removed by filtering: {data_tensor.shape[0] - filtered_data.shape[0]}")
        filt_cls = torch.argmax(filtered_label, dim=1)
        unique_f, counts_f = torch.unique(filt_cls, return_counts=True)
        for u, c in zip(unique_f, counts_f):
            print(f"[FILT] Class {int(u)} : {int(c)} samples")
        print("===============================================\n")
    else:
        filtered_data = data_tensor
        filtered_label = logits_all

    # ----- 저장 -----
    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (A) augmentation only
    torch.save(data_tensor.cpu(), out_dir / 'augmented_data.pt')
    torch.save(logits_all.cpu(), out_dir / 'augmented_label.pt')
    print(f"[INFO] Saved augmented tensors to {out_dir / 'augmented_data.pt'} and {out_dir / 'augmented_label.pt'}")

    # (B) filtering 적용 버전 (또는 동일)
    torch.save(filtered_data.cpu(), out_dir / 'generated_data.pt')
    torch.save(filtered_label.cpu(), out_dir / 'generated_label.pt')
    print(f"[INFO] Saved final tensors to {out_dir / 'generated_data.pt'} and {out_dir / 'generated_label.pt'}")


if __name__ == '__main__':
    main()