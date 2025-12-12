import os
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils_cifar import *  # DeconvNet, (optionally get_vggmodel 등)


# -----------------------------
# 0. Device 설정
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# 1. VGG fc1 feature 추출
# -----------------------------
def get_output(vgg_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    VGG 계열 victim에서 features → avgpool → classifier[0] 까지 통과시킨 fc1 특징 추출.
    x: (B, 3, 32, 32)
    return: (B, D)
    """
    with torch.no_grad():
        feats = vgg_model.features(x)
        pooled = vgg_model.avgpool(feats)
        pooled = pooled.view(pooled.size(0), -1)
        fc1 = vgg_model.classifier[0](pooled)
    return fc1


# -----------------------------
# 2. 모델 초기화 (DeconvNet)
# -----------------------------
def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -----------------------------
# 3. Victim 로드
# -----------------------------
def load_victim_model(path: str, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Victim model not found: {path}")
    victim_builder = get_vggmodel
    try:
        victim_obj = torch.load(path, map_location=device)
        if isinstance(victim_obj, dict):
            victim_model = victim_builder()
            victim_model.load_state_dict(victim_obj)
        else:
            victim_model = victim_obj
    except Exception:
        # fallback: try loading to CPU then move
        victim_model = torch.load(path, map_location='cpu')

    victim_model = victim_model.to(device)
    victim_model.eval()
    return victim_model


# -----------------------------
# 4. generated_data → reverse_data (feat, img) 생성
# -----------------------------
def build_reverse_data(
    generated_data_path: str,
    victim_model: nn.Module,
    device,
    reverse_data_path: str,
    reverse_label_path: str,
    batch_size: int = 128,
):
    """
    generated_data.pt (stealing set 이미지)에서
    victim fc1 feature를 계산해 (feature, image) 쌍으로 저장.
    """
    gen_path = Path(generated_data_path)
    if not gen_path.exists():
        raise FileNotFoundError(f"generated_data.pt not found: {gen_path}")

    imgs = torch.load(gen_path, map_location="cpu")  # (N, 3, 32, 32)
    print(f"[INFO] Loaded generated_data: {imgs.shape}")

    dataset = TensorDataset(imgs, imgs)  # label dummy
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feat_list = []
    img_list = []

    victim_model.eval()
    with torch.no_grad():
        for batch_imgs, _ in loader:
            batch_imgs = batch_imgs.to(device)
            fc1 = get_output(victim_model, batch_imgs)  # (B, D)
            feat_list.append(fc1.cpu())
            img_list.append(batch_imgs.cpu())

    feats = torch.cat(feat_list, dim=0)  # (N, D)
    imgs = torch.cat(img_list, dim=0)    # (N, 3, 32, 32)
    print(f"[INFO] Built reverse tensors: feats={feats.shape}, imgs={imgs.shape}")

    torch.save(imgs, reverse_data_path)
    torch.save(feats, reverse_label_path)
    print(f"[INFO] Saved reverse_data to {reverse_data_path}")
    print(f"[INFO] Saved reverse_label to {reverse_label_path}")
    return feats, imgs


# -----------------------------
# 5. reverse_data / reverse_label 로드 & 정리
# -----------------------------
def load_reverse_dataset(
    reverse_data_path: str,
    reverse_label_path: str,
):
    data = torch.load(reverse_data_path, map_location="cpu")
    label = torch.load(reverse_label_path, map_location="cpu")
    print('reverse shapes: ', data.shape, label.shape)
    # data: (N, 3, 32, 32), label: (N, D) 인지 확인
    if data.ndim == 4 and label.ndim == 2:
        imgs, feats = data, label
    elif data.ndim == 2 and label.ndim == 4:
        feats, imgs = data, label
    else:
        raise RuntimeError(
            f"Unexpected shapes: reverse_data={tuple(data.shape)}, "
            f"reverse_label={tuple(label.shape)}"
        )

    print(f"[INFO] Loaded reverse dataset: imgs={imgs.shape}, feats={feats.shape}")
    dataset = TensorDataset(feats, imgs)  # (feature, image)
    return dataset


# -----------------------------
# 6. Inversion 모델 학습
# -----------------------------
def train_inversion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    num_epochs: int = 200,
    lr: float = 5e-4,
    save_path: str = "inversion.pt",
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for feats, imgs in train_loader:
            feats, imgs = feats.to(device), imgs.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, imgs) * 10.0  # 원 코드 스케일 유지
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, imgs in val_loader:
                feats, imgs = feats.to(device), imgs.to(device)
                outputs = model(feats)
                loss = criterion(outputs, imgs) * 10.0
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"[INFO] Saved best inversion model to {save_path} (Val Loss={best_val_loss:.4f})")

    return best_val_loss


# -----------------------------
# 7. Inversion 평가 (reverse_dataset)
# -----------------------------
def evaluate_on_reverse(
    model: nn.Module,
    victim_model: nn.Module,
    dataset: TensorDataset,
    device,
    batch_size: int = 128,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mse_loss = nn.MSELoss()
    total_mse = 0.0
    total = 0
    agree = 0

    model.eval()
    victim_model.eval()

    with torch.no_grad():
        for feats, imgs in loader:
            feats, imgs = feats.to(device), imgs.to(device)
            recon = model(feats)

            mse = mse_loss(recon, imgs)
            total_mse += mse.item() * imgs.size(0)
            total += imgs.size(0)

            # agreement: victim(img) vs victim(recon)
            pred_orig = victim_model(imgs).argmax(dim=1)
            pred_recon = victim_model(recon).argmax(dim=1)
            agree += (pred_orig == pred_recon).sum().item()

    avg_mse = total_mse / total
    agreement = 100.0 * agree / total
    print(f"[EVAL-Reverse] Avg MSE: {avg_mse:.6f}, Agreement: {agreement:.2f}%")
    return avg_mse, agreement


# -----------------------------
# 8. CIFAR10 testset 위에서 평가
# -----------------------------
def evaluate_on_cifar_test(
    model: nn.Module,
    victim_model: nn.Module,
    device,
    batch_size: int = 128,
    num_visual: int = 4,
):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    mse_loss = nn.MSELoss()
    total_mse = 0.0
    total = 0
    agree = 0

    model.eval()
    victim_model.eval()

    vis_orig = []
    vis_recon = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = get_output(victim_model, imgs)
            recon = model(feats)

            mse = mse_loss(recon, imgs)
            total_mse += mse.item() * imgs.size(0)
            total += imgs.size(0)

            pred_orig = victim_model(imgs).argmax(dim=1)
            pred_recon = victim_model(recon).argmax(dim=1)
            agree += (pred_orig == pred_recon).sum().item()

            # 시각화용 샘플 몇 개 모으기
            if len(vis_orig) < num_visual:
                for i in range(min(num_visual - len(vis_orig), imgs.size(0))):
                    vis_orig.append(imgs[i].cpu())
                    vis_recon.append(recon[i].cpu())

    avg_mse = total_mse / total
    agreement = 100.0 * agree / total
    print(f"[EVAL-CIFAR10] Avg MSE: {avg_mse:.6f}, Agreement: {agreement:.2f}%")

    # 시각화
    if num_visual > 0 and len(vis_orig) > 0:
        fig, axes = plt.subplots(len(vis_orig), 2, figsize=(4, 2 * len(vis_orig)))
        for i in range(len(vis_orig)):
            orig = vis_orig[i].permute(1, 2, 0).numpy()
            recon = vis_recon[i].permute(1, 2, 0).numpy()

            axes[i, 0].imshow(orig)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon)
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

    return avg_mse, agreement


# -----------------------------
# 9. main
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model inversion attack (CIFAR10, VGG victim)")
    parser.add_argument("--victim-path", type=str, default="./results/vgg16_victim/vgg16_victim_cifar10_state.pt",
                        help="Path to victim VGG model (full torch module).")
    parser.add_argument("--generated-data", type=str, default="./stealing_set/SD1.5/vgg16_False/generated_data.pt",
                        help="Path to stealing set images (for building reverse_data).")
    parser.add_argument("--reverse-data", type=str, default="./stealing_set/SD1.5/vgg16_False/reverse_data.pt",
                        help="Path to reverse_data (images) tensor.")
    parser.add_argument("--reverse-label", type=str, default="./stealing_set/SD1.5/vgg16_False/reverse_label.pt",
                        help="Path to reverse_label (features) tensor.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-path", type=str, default="./stealing_set/SD1.5/vgg16_False/inversion.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Using device: {device}")

    # 1) victim 로드
    victim_model = load_victim_model(args.victim_path, device)

    # 2) reverse_data / reverse_label 준비
    rev_data_path = Path(args.reverse_data)
    rev_label_path = Path(args.reverse_label)

    if rev_data_path.exists() and rev_label_path.exists():
        print("[INFO] Found existing reverse_data / reverse_label. Loading...")
        reverse_dataset = load_reverse_dataset(rev_data_path, rev_label_path)
    else:
        print("[INFO] reverse_data / reverse_label not found. Building from generated_data...")
        feats, imgs = build_reverse_data(
            args.generated_data,
            victim_model,
            device,
            reverse_data_path=str(rev_data_path),
            reverse_label_path=str(rev_label_path),
            batch_size=args.batch_size,
        )
        reverse_dataset = TensorDataset(feats, imgs)

    # 3) train/val split
    val_size = max(1, int(len(reverse_dataset) * args.val_ratio))
    train_size = len(reverse_dataset) - val_size
    train_dataset, val_dataset = random_split(reverse_dataset, [train_size, val_size])
    print(f"[INFO] Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4) inversion 모델 정의
    model = DeconvNet()
    model.apply(init_weights)

    # 5) 학습
    best_val = train_inversion_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.out_path,
    )

    # best 모델 로드
    state = torch.load(args.out_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    # 6) reverse_dataset 위 평가
    evaluate_on_reverse(model, victim_model, reverse_dataset, device, batch_size=args.batch_size)

    # 7) CIFAR10 testset 위 평가 + 시각화
    evaluate_on_cifar_test(model, victim_model, device, batch_size=args.batch_size, num_visual=4)

    print(f"[DONE] Best Val Loss: {best_val:.4f}")


if __name__ == "__main__":
    main()