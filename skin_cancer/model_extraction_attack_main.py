import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.datasets as dset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

from utils_skin import get_vggmodel


# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(
        description="Model Extraction Attack on Skin Cancer (GAN-based stealing set)"
    )
    parser.add_argument(
        "--victim-path",
        type=str,
        default="./results/vgg16_victim/vgg16_victim_skin_state.pt",
        help="Path to victim model checkpoint (full model or state_dict).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./stealing_set_GAN/filtered_data.pt",
        help="Path to generated/stealing images tensor (.pt).",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="./stealing_set_GAN/filtered_label.pt",
        help="Path to generated/stealing labels tensor (.pt).",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["augmented", "filtered"],
        default="filtered",
        help="Type of stealing set: 'augmented' or 'filtered' (used for results dir name).",
    )
    parser.add_argument(
        "--val-root",
        type=str,
        default="./melanoma_cancer_dataset/test",
        help="Root directory of validation/test set (ImageFolder).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for stealing set loader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for student (steal) model.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Number of batches between logging training loss.",
    )
    return parser.parse_args()


# Device
# device = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
# )
device = 'cpu'
print(f"[INFO] Using device: {device}")


# Data Transform
IMAGE_SIZE = 64
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Victim Model Loader
def load_victim_model(path: str) -> nn.Module:
    """
    Victim 모델 로더:
    - torch.save(model) 로 저장된 전체 모델이든,
    - model.state_dict() 만 저장된 것이든 모두 처리.
    """
    print(f"[INFO] Loading victim model from: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        victim = obj
    else:
        # state_dict
        victim = get_vggmodel()
        victim.load_state_dict(obj)

    victim.to(device)
    victim.eval()
    return victim

# Evaluation (loss, accuracy, agreement)
def evaluate(model, victim_model, loader, criterion_eval):
    """
    - model: student / stealing model (nn.Module)
    - victim_model: victim (nn.Module)
    - loader: val/test dataloader
    - criterion_eval: CrossEntropyLoss 등
    return: (val_loss, val_accuracy, agreement)
    """
    model.eval()
    victim_model.eval()

    val_loss = 0.0
    total = 0
    correct = 0
    agree = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)         # (B, 2)
            victim_outputs = victim_model(images)

            loss = criterion_eval(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            _, victim_preds = torch.max(victim_outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            agree += (preds == victim_preds).sum().item()

    avg_loss = val_loss / len(loader)
    accuracy = 100.0 * correct / total
    agreement = 100.0 * agree / total
    return avg_loss, accuracy, agreement


# Plotting helpers
def plot_loss(train_losses, val_losses, save_path: Path):
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_acc_agreement(val_accs, agreements, save_path: Path):
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(val_accs) + 1)
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.plot(epochs, agreements, label="Agreement")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.title("Accuracy & Agreement")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# Main Training Loop
def main():
    args = get_args()
    print(f"[INFO] Args: {args}")

    # ./results/vgg16_extraction_GAN_{augmented|filtered}
    results_dir = Path(f"./results/vgg16_extraction_GAN_{args.data_type}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {results_dir}")

    # 1) Validation/Test set (real skin cancer images)
    val_dataset = dset.ImageFolder(root=args.val_root, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    print(f"[INFO] Validation set size: {len(val_dataset)}")

    # 2) Victim model
    victim_model = load_victim_model(args.victim_path)

    # 3) Stealing set (GAN generated)
    generate_img = torch.load(args.data_path)
    generate_label = torch.load(args.label_path)
    print("[INFO] Loaded stealing set tensors:")
    print("       images:", generate_img.shape, "labels:", generate_label.shape)

    torch_dataset = Data.TensorDataset(generate_img, generate_label)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # 4) Student / Stealing model
    skin_model = get_vggmodel()
    skin_model.to(device)

    # knowledge distilation: MSE (soft labels / one-hot labels)
    criterion_train = nn.MSELoss()
    # eval: CrossEntropyLoss (hard gt label)
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        skin_model.parameters(), lr=args.lr, momentum=0.9
    )

    num_epochs = args.epochs

    train_losses = []
    val_losses = []
    val_accuracies = []
    agreements = []

    best_agreement = -1.0
    best_epoch = -1

    # CSV logging
    csv_path = results_dir / "metrics_log.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_accuracy", "agreement"]
        )

    # Training
    for epoch in range(1, num_epochs + 1):
        skin_model.train()
        running_loss = 0.0

        pbar = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            desc=f"Epoch {epoch}/{num_epochs}",
        )

        for batch_idx, (inputs, labels) in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = skin_model(inputs)

            loss = criterion_train(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                pbar.set_postfix(
                    {"batch_loss": f"{loss.item():.4f}"}
                )

        avg_train_loss = running_loss / len(loader)
        train_losses.append(avg_train_loss)

        # Validation / Test
        val_loss, val_acc, agreement = evaluate(
            skin_model, victim_model, val_loader, criterion_eval
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        agreements.append(agreement)

        # print epoch summary
        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Agreement: {agreement:.2f}%"
        )

        # CSV logging append
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, avg_train_loss, val_loss, val_acc, agreement]
            )

        # Best model (by agreement) renewal
        if agreement > best_agreement:
            best_agreement = agreement
            best_epoch = epoch
            torch.save(
                skin_model.state_dict(),
                results_dir / "steal_model_best_state.pt",
            )
            print(
                f"[INFO] New best agreement: {best_agreement:.2f}% at epoch {best_epoch}"
            )

    # save final model
    torch.save(skin_model, results_dir / "steal_model_last.pt")
    print(
        f"[INFO] Training complete. Final model saved to "
        f"{results_dir / 'steal_model_last.pt'}"
    )

    # save plots
    plot_loss(
        train_losses,
        val_losses,
        results_dir / "loss_curve.png",
    )
    plot_acc_agreement(
        val_accuracies,
        agreements,
        results_dir / "accuracy_agreement.png",
    )
    print("[INFO] Plots saved (loss_curve.png, accuracy_agreement.png)")

    # best summary txt
    summary_path = results_dir / "best_summary.txt"
    with summary_path.open("w") as f:
        f.write(
            f"Best epoch: {best_epoch}\n"
            f"Best agreement: {best_agreement:.4f}%\n"
            f"Final epoch: {num_epochs}\n"
            f"Final val accuracy: {val_accuracies[-1]:.4f}%\n"
            f"Final val loss: {val_losses[-1]:.4f}\n"
        )
    print("[INFO] Best summary saved to best_summary.txt")


if __name__ == "__main__":
    main()