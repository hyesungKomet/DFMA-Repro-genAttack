import os, sys
import random
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

import gradio as gr
from PIL import Image
import numpy as np

# Paths 
VICTIM_PATH = "./cifar10/results/vgg16_victim/vgg16_victim_cifar10_state.pt"
EXTRACTED_PATH = "./cifar10/results/vgg16_extraction_SD1.5_True/vgg16_extracted_cifar10_state.pt"
MIA_PATH = "./cifar10/results/mia_model/meta_attacker_SD1.5true.pt"
INVERSION_PATH = "./cifar10/results/inversion_model/cifar_inversion_model.pth"

# Synthetic png folder
SYNTH_ROOT = Path("./cifar10/generated_data_SD1.5")

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"[INFO] Using device: {device}")

# CIFAR-10 classes
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CIFAR10_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Transforms (resize 32 + ImageNet normalize)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_for_models = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    norm,
])

# for display / inversion pipeline (keep 0~1 tensor)
transform_raw01 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def pil_to_model_tensor(pil: Image.Image) -> torch.Tensor:
    """(1,3,32,32) normalized tensor for victim/extracted forward."""
    return transform_for_models(pil).unsqueeze(0)


def pil_to_raw01_tensor(pil: Image.Image) -> torch.Tensor:
    """(1,3,32,32) tensor in [0,1] (no normalize)."""
    return transform_raw01(pil).unsqueeze(0)


def raw01_to_model_tensor(x01: torch.Tensor) -> torch.Tensor:
    """Convert (1,3,32,32) [0,1] -> normalized for victim/extracted."""
    x = x01.clone()
    for c in range(3):
        x[:, c] = (x[:, c] - norm.mean[c]) / norm.std[c]
    return x


def tensor01_to_pil(x01: torch.Tensor) -> Image.Image:
    """(1,3,32,32) or (3,32,32) in [0,1] -> PIL"""
    if x01.dim() == 4:
        x01 = x01[0]
    x01 = x01.detach().cpu().clamp(0, 1)
    arr = (x01.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# Models
def build_vgg16_cifar10() -> nn.Module:
    # Match utils_cifar.get_vggmodel(): vgg16(pretrained=True) + classifier[6]=Linear(...,10)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)
    return model


def load_vgg_state(path: str) -> nn.Module:
    # victim/extracted are typically state_dict. `weights_only` default True is fine.
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        m = build_vgg16_cifar10()
        m.load_state_dict(obj)
    else:
        m = obj
    m.to(device).eval()
    return m


# MIA MetaAttack fallback 
class MetaAttack(nn.Module):
    """
    Meta-classifier for membership inference attack.
    Takes concatenated model outputs and true label as input,
    outputs probability of being a training member.
    """
    def __init__(self, input_dim=11, hidden_dims=[256, 128, 64]):
        """
        Args:
            input_dim: Input dimension (num_classes + 1 for CIFAR-10: 10 + 1 = 11)
            hidden_dims: List of hidden layer dimensions
        """
        super(MetaAttack, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 11)
               First 10 dims: model output probabilities/logits
               Last 1 dim: true label
        
        Returns:
            Output tensor of shape (batch_size, 1) with membership probability
        """
        return self.model(x)




def load_mia_model(path: str) -> nn.Module:
    """
    Load MIA meta-attacker checkpoint (pickled full model).
    """

    # 🔥 핵심: cifar10/self_model.py 를
    # pickle 이 기대하는 top-level 모듈명 `self_model`로 인식시키기
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cifar10_dir = os.path.join(repo_root, "cifar10")
    if cifar10_dir not in sys.path:
        sys.path.insert(0, cifar10_dir)

    # 이제 이 import가 성공해야 함
    from self_model import MetaAttack as SavedMetaAttack

    # (권장) torch safe globals 등록
    try:
        torch.serialization.add_safe_globals([SavedMetaAttack])
    except Exception:
        pass

    # ✅ 반드시 weights_only=False
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        m = obj
    elif isinstance(obj, dict):
        m = SavedMetaAttack(input_dim=11)
        m.load_state_dict(obj)
    else:
        raise TypeError(f"Unsupported MIA checkpoint type: {type(obj)}")

    m.to(device).eval()
    return m

# Inversion net (from model_inversion_attack_main3.py)
def get_features_vgg16(victim_model: nn.Module, x_normed: torch.Tensor) -> torch.Tensor:
    """features -> avgpool -> flatten -> classifier[0] -> classifier[1]"""
    with torch.no_grad():
        feat = victim_model.features(x_normed)
        feat = victim_model.avgpool(feat)
        feat = feat.view(feat.size(0), -1)  # 25088
        feat = victim_model.classifier[0](feat)  # fc1 -> 4096
        feat = victim_model.classifier[1](feat)  # ReLU
    return feat


class CIFARInversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4096, 256 * 2 * 2)
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),  # [-1,1]
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 256, 2, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


def load_inversion_model(path: str) -> nn.Module:
    # Inversion model may be saved as full pickled model.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            m = obj
        elif isinstance(obj, dict):
            m = CIFARInversionNet()
            m.load_state_dict(obj)
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
        m.to(device).eval()
        return m
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        m = CIFARInversionNet()
        m.load_state_dict(obj)
        m.to(device).eval()
        return m


# Load everything once
print("[INFO] Loading victim/extracted/MIA/inversion models...")
victim_model = load_vgg_state(VICTIM_PATH)
extracted_model = load_vgg_state(EXTRACTED_PATH)
mia_model = load_mia_model(MIA_PATH)
inversion_model = load_inversion_model(INVERSION_PATH)
print("[INFO] Models loaded.")

# CIFAR-10 dataset (for "real" samples)
cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)


def sample_real_cifar() -> Tuple[Image.Image, int, str]:
    idx = random.randint(0, len(cifar_test) - 1)
    pil, y = cifar_test[idx]
    return pil.convert("RGB"), y, "CIFAR-10 test"


def sample_synth_png() -> Tuple[Image.Image, int, str]:
    if not SYNTH_ROOT.exists():
        raise FileNotFoundError(f"Synthetic folder not found: {SYNTH_ROOT}")

    class_dirs = [p for p in SYNTH_ROOT.iterdir() if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders in: {SYNTH_ROOT}")

    cls_dir = random.choice(class_dirs)
    cls_name = cls_dir.name
    pngs = list(cls_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No png in: {cls_dir}")

    img_path = random.choice(pngs)
    pil = Image.open(img_path).convert("RGB")

    if cls_name in CLASS_TO_IDX:
        y = CLASS_TO_IDX[cls_name]
    else:
        try:
            y = int(cls_name)
        except Exception:
            y = 0

    return pil, y, f"Synthetic ({img_path})"


# Inference helpers
@torch.no_grad()
def predict_probs(model: nn.Module, x_normed: torch.Tensor) -> np.ndarray:
    logits = model(x_normed.to(device))
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return probs


def topk_dict(probs: np.ndarray, k=5) -> Dict[str, float]:
    idxs = probs.argsort()[::-1][:k]
    return {IDX_TO_CLASS[int(i)]: float(probs[i]) for i in idxs}


def mia_score_from_victim_output(victim_probs: np.ndarray, true_label: int) -> float:
    """Attack input format: [10 probs, true_label] -> sigmoid prob(member)."""
    feat = torch.tensor(
        np.concatenate([victim_probs.astype(np.float32), np.array([true_label], dtype=np.float32)]),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        p = mia_model(feat).squeeze().item()
    return float(p)


def run_inversion(pil: Image.Image):
    """Reconstruct image via inversion model and compute metrics."""
    x01 = pil_to_raw01_tensor(pil).to(device)  # [0,1]
    x_norm = raw01_to_model_tensor(x01).to(device)

    feat = get_features_vgg16(victim_model, x_norm)
    recon_m11 = inversion_model(feat)  # [-1,1]

    recon01 = (recon_m11 + 1.0) / 2.0
    recon01 = recon01.clamp(0, 1)

    mse = torch.mean((recon01 - x01) ** 2).item()

    recon_norm = raw01_to_model_tensor(recon01).to(device)
    real_probs = predict_probs(victim_model, x_norm)
    recon_probs = predict_probs(victim_model, recon_norm)
    real_pred = int(np.argmax(real_probs))
    recon_pred = int(np.argmax(recon_probs))
    same = real_pred == recon_pred

    msg = f"Victim pred(real)={IDX_TO_CLASS[real_pred]}, pred(recon)={IDX_TO_CLASS[recon_pred]}"
    return tensor01_to_pil(recon01), float(mse), bool(same), msg


# Gradio callbacks
def load_sample(source: str):
    if source == "Real CIFAR-10 test image":
        pil, y, meta = sample_real_cifar()
    else:
        pil, y, meta = sample_synth_png()
    return pil, CIFAR10_CLASSES[y], meta


def analyze(pil: Image.Image, label_name: str):
    if pil is None:
        return None, None, None, None, None, None, None

    true_label = CLASS_TO_IDX.get(label_name, 0)

    x_norm = pil_to_model_tensor(pil).to(device)
    victim_probs = predict_probs(victim_model, x_norm)
    extracted_probs = predict_probs(extracted_model, x_norm)

    victim_top5 = topk_dict(victim_probs, 5)
    extracted_top5 = topk_dict(extracted_probs, 5)

    v_pred = int(np.argmax(victim_probs))
    e_pred = int(np.argmax(extracted_probs))
    agree = v_pred == e_pred

    mia_p = mia_score_from_victim_output(victim_probs, true_label)
    mia_pred = 1 if mia_p >= 0.5 else 0

    recon_pil, mse, same_pred, inv_msg = run_inversion(pil)

    summary = (
        f"**Victim Top-1:** {IDX_TO_CLASS[v_pred]} ({victim_probs[v_pred]:.3f})\n\n"
        f"**Extracted Top-1:** {IDX_TO_CLASS[e_pred]} ({extracted_probs[e_pred]:.3f})\n\n"
        f"**Victim↔Extracted Agreement:** {agree}\n\n"
        f"**MIA(member prob):** {mia_p:.3f} → pred={mia_pred}\n\n"
        f"**Inversion MSE:** {mse:.6f}\n\n"
        f"**Inversion victim-pred match:** {same_pred}  \n{inv_msg}"
    )

    return victim_top5, extracted_top5, agree, mia_p, mia_pred, recon_pil, summary


# UI
with gr.Blocks(title="CIFAR-10 Privacy Attacks Demo") as demo:
    gr.Markdown("# CIFAR-10 Privacy Attacks Demo (Victim / Extraction / MIA / Inversion)")

    with gr.Row():
        with gr.Column():
            source = gr.Radio(
                choices=["Real CIFAR-10 test image", "Synthetic (SD1.5) png"],
                value="Real CIFAR-10 test image",
                label="Sample source",
            )
            btn_sample = gr.Button("Load random sample")
            img_in = gr.Image(type="pil", label="Input image (you can also upload)")

            label_dd = gr.Dropdown(choices=CIFAR10_CLASSES, value="airplane", label="True label (for MIA input)")
            meta_info = gr.Textbox(label="Sample info", interactive=False)

        with gr.Column():
            victim_top5_out = gr.Label(num_top_classes=5, label="Victim model top-5")
            extracted_top5_out = gr.Label(num_top_classes=5, label="Extracted model top-5")
            agree_out = gr.Checkbox(label="Victim ↔ Extracted Top-1 agreement", interactive=False)

            mia_prob_out = gr.Number(label="MIA: P(member)", interactive=False)
            mia_pred_out = gr.Number(label="MIA: predicted class (0=non-member, 1=member)", interactive=False)

            recon_out = gr.Image(type="pil", label="Inversion reconstruction")
            summary_out = gr.Markdown()

    btn_analyze = gr.Button("Run analysis")

    btn_sample.click(fn=load_sample, inputs=[source], outputs=[img_in, label_dd, meta_info])

    btn_analyze.click(
        fn=analyze,
        inputs=[img_in, label_dd],
        outputs=[victim_top5_out, extracted_top5_out, agree_out, mia_prob_out, mia_pred_out, recon_out, summary_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)