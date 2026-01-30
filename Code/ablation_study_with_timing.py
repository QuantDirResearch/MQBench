import argparse
import gc
import io
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization


# ----------------------------
# Defaults (you can override via CLI)
# ----------------------------
DEFAULT_SEED = 42
DEFAULT_NUM_CALIB = 1024
DEFAULT_NUM_EVAL = 50000
DEFAULT_CALIB_BS = 32
DEFAULT_EVAL_BS = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_N_SELECT = 7
DEFAULT_WARMUP_STEPS = 5

# Model-agnostic early/late by output feature-map resolution
EARLY_MIN_AREA = 28 * 28
LATE_MAX_AREA = 14 * 14

# Per-layer fitting limits (offline cost control)
MAX_CALIB_BATCHES_CONV = 25
MAX_CALIB_BATCHES_BN = 20

# If you want classifier to be eligible in LATE filtering (selection only)
INCLUDE_CLASSIFIER_IN_LATE = True


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Model factory
# ----------------------------
def build_fp32_model(model_name: str) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if model_name == "resnet34":
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    if model_name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if model_name == "mobilenet_v2":
        return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    if model_name == "vgg16":
        return models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    if model_name == "vgg19":
        return models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    if model_name == "regnetx800mf":
        # Works across torchvision versions
        try:
            w = models.RegNet_X_800MF_Weights.IMAGENET1K_V1
            return models.regnet_x_800mf(weights=w)
        except Exception:
            return models.regnet_x_800mf(pretrained=True)

    raise ValueError(
        "Unknown model_name. Choose one of: "
        "resnet18, resnet34, resnet50, regnetx800mf, mobilenet_v2, vgg16, vgg19"
    )


# ----------------------------
# CSV paths (defaults for your machine; override with --csv-path if needed)
# ----------------------------
def default_csv_paths() -> Dict[str, str]:
    base = "/Users/shailasharmin/Desktop/RA/ptq_metrics"
    return {
        "resnet18": os.path.join(base, "BeforeCorrection", "instability_metrics_resnet18.csv"),
        "resnet34": os.path.join(base, "instability_metrics_resnet34.csv"),
        "resnet50": os.path.join(base, "instability_metrics_resnet50.csv"),
        "mobilenet_v2": os.path.join(base, "instability_metrics_mobilenetv2.csv"),
        "vgg16": os.path.join(base, "instability_metrics_vgg16.csv"),
        "vgg19": os.path.join(base, "instability_metrics_vgg19.csv"),
        "regnetx800mf": os.path.join(base, "instability_metrics_regnet800.csv"),
    }


# ----------------------------
# Data loaders
# ----------------------------
def get_loaders(
    imagenet_root: str,
    num_calib: int,
    num_eval: int,
    seed: int,
    calib_bs: int,
    eval_bs: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    set_seed(seed)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    ds = datasets.ImageFolder(root=imagenet_root, transform=tfm)
    n = len(ds)

    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    calib_idx = idx[: min(num_calib, n)]
    start = min(num_calib, n)
    eval_idx = idx[start: min(start + num_eval, n)]

    calib_ds = Subset(ds, calib_idx)
    eval_ds = Subset(ds, eval_idx)

    pin = torch.cuda.is_available()
    persistent = (num_workers > 0)

    calib_loader = DataLoader(
        calib_ds,
        batch_size=calib_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return calib_loader, eval_loader


# ----------------------------
# Evaluation (accuracy + inference time over full eval_loader)
# ----------------------------
@torch.no_grad()
def evaluate_topk(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    top1 = 0
    top5 = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        _, pred = out.topk(5, dim=1, largest=True, sorted=True)
        total += y.size(0)
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()
    return 100.0 * top1 / max(total, 1), 100.0 * top5 / max(total, 1)


@torch.no_grad()
def measure_inference_time_seconds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> float:
    model.eval()

    # warmup on real batches (does NOT affect accuracy because we time separately)
    it = iter(loader)
    for _ in range(warmup_steps):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return time.perf_counter() - t0


# ----------------------------
# MQBench PTQ helper
# ----------------------------
@torch.no_grad()
def calibrate_and_quantize(q_model: nn.Module, calib_loader: DataLoader, device: torch.device):
    enable_calibration(q_model)
    q_model.eval()
    for x, _ in calib_loader:
        _ = q_model(x.to(device, non_blocking=True))
    enable_quantization(q_model)


def default_int4w8_extra_config() -> dict:
    # You can tune observers/symmetry as needed; this is a stable baseline.
    return {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "MinMaxObserver",
            "w_fakequantize": "FixedFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {"bit": 4, "symmetry": False, "per_channel": True, "pot_scale": False},
            "a_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False},
        }
    }


# ----------------------------
# CSV utilities
# ----------------------------
def load_layer_names_from_csv(csv_path: str) -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "layer" not in df.columns:
        raise ValueError(f"CSV must contain a 'layer' column. Found: {list(df.columns)}")

    layers = (
        df["layer"].astype(str).str.strip()
        .replace({"": None, "nan": None, "None": None})
        .dropna().unique().tolist()
    )
    if not layers:
        raise ValueError("No valid layer names in CSV column 'layer'.")
    return layers


def filter_layers_in_model(layer_names: List[str], model: nn.Module) -> List[str]:
    mods = dict(model.named_modules())
    return [n for n in layer_names if n in mods]


def sample_layers(layer_names: List[str], n_select: int, seed: int) -> List[str]:
    if len(layer_names) <= 0:
        return []
    if len(layer_names) < n_select:
        # If you ask for more than available, take all (safer than crashing mid-run)
        return list(layer_names)
    rnd = random.Random(seed)
    return rnd.sample(layer_names, n_select)


def bucket_layers_by_module_type(model: nn.Module, layer_names: List[str]) -> Dict[str, List[str]]:
    mods = dict(model.named_modules())
    buckets = {"conv": [], "bn": [], "relu": [], "other": []}
    for name in layer_names:
        m = mods.get(name, None)
        if m is None:
            buckets["other"].append(name)
            continue
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            buckets["conv"].append(name)
        elif isinstance(m, nn.BatchNorm2d):
            buckets["bn"].append(name)
        elif isinstance(m, nn.ReLU):
            buckets["relu"].append(name)
        else:
            buckets["other"].append(name)
    return buckets


# ----------------------------
# EARLY/LATE selection via output resolution (model-agnostic)
# ----------------------------
@torch.no_grad()
def build_resolution_map(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
    include_types=(nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear),
) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """
    name -> (H, W) for 4D outputs; (None, None) otherwise
    """
    model.eval()
    mods = dict(model.named_modules())
    res_map: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_m, _i, o):
            out = o[0] if isinstance(o, (tuple, list)) else o
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                res_map[name] = (int(out.size(2)), int(out.size(3)))
            else:
                res_map[name] = (None, None)
        return hook

    for name, m in mods.items():
        if isinstance(m, include_types):
            hooks.append(m.register_forward_hook(make_hook(name)))

    x = torch.randn(1, 3, input_size, input_size, device=device)
    _ = model(x)

    for h in hooks:
        h.remove()

    return res_map


def filter_layers_early_late(
    layer_names: List[str],
    model: nn.Module,
    res_map: Dict[str, Tuple[Optional[int], Optional[int]]],
    mode: str,
    include_classifier_in_late: bool,
) -> List[str]:
    mode = mode.upper().strip()
    mods = dict(model.named_modules())

    valid = [n for n in layer_names if (n in mods and n in res_map)]
    if mode == "RANDOM":
        return valid

    early, late = [], []
    for n in valid:
        h, w = res_map.get(n, (None, None))
        if h is None or w is None:
            # classifier/head (non-4D output)
            if include_classifier_in_late and (n.startswith("classifier") or n == "fc"):
                late.append(n)
            continue

        area = h * w
        if area >= EARLY_MIN_AREA:
            early.append(n)
        if area <= LATE_MAX_AREA:
            late.append(n)

    if mode == "EARLY":
        return early
    if mode == "LATE":
        return late
    raise ValueError("mode must be one of: RANDOM, EARLY, LATE")


def select_layers_from_csv(
    csv_path: str,
    fp32_model: nn.Module,
    device: torch.device,
    mode: str,
    n_select: int,
    seed: int,
) -> List[str]:
    layers = load_layer_names_from_csv(csv_path)
    layers = filter_layers_in_model(layers, fp32_model)

    res_map = build_resolution_map(fp32_model, device=device, input_size=224)
    layers = filter_layers_early_late(
        layers, model=fp32_model, res_map=res_map,
        mode=mode, include_classifier_in_late=INCLUDE_CLASSIFIER_IN_LATE
    )
    return sample_layers(layers, n_select=n_select, seed=seed)


# ----------------------------
# QuantDIR module (renamed from QuantDIRFilter, same math)
# ----------------------------
class QuantDIRFilter(nn.Module):
    def __init__(self, num_channels: int, gain: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.register_buffer("K", gain.view(-1))
        self.register_buffer("alpha", alpha.view(-1))
        self.register_buffer("beta", beta.view(-1))

    def forward(self, y_q: torch.Tensor) -> torch.Tensor:
        if y_q.dim() == 4:
            K = self.K.view(1, -1, 1, 1)
            alpha = self.alpha.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        else:
            K = self.K.view(1, -1)
            alpha = self.alpha.view(1, -1)
            beta = self.beta.view(1, -1)

        corrected = alpha * y_q + beta
        estimate = K * y_q + (1.0 - K) * corrected
        return estimate


class _RunningStats:
    """
    Streaming stats (CPU float64 vectors):
      sum_fp, sum_q, sumsq_fp, sumsq_q, sum_fpq, count
    """
    def __init__(self, C: int, eps: float = 1e-6):
        self.eps = eps
        self.count = 0
        self.sum_fp = torch.zeros(C, dtype=torch.float64)
        self.sum_q = torch.zeros(C, dtype=torch.float64)
        self.sumsq_fp = torch.zeros(C, dtype=torch.float64)
        self.sumsq_q = torch.zeros(C, dtype=torch.float64)
        self.sum_fpq = torch.zeros(C, dtype=torch.float64)

    def update(self, y_fp: torch.Tensor, y_q: torch.Tensor):
        if y_fp.dim() == 4:
            n = y_fp.size(0) * y_fp.size(2) * y_fp.size(3)
            s_fp = y_fp.sum(dim=(0, 2, 3))
            s_q = y_q.sum(dim=(0, 2, 3))
            ss_fp = (y_fp * y_fp).sum(dim=(0, 2, 3))
            ss_q = (y_q * y_q).sum(dim=(0, 2, 3))
            s_fpq = (y_fp * y_q).sum(dim=(0, 2, 3))
        else:
            n = y_fp.size(0)
            s_fp = y_fp.sum(dim=0)
            s_q = y_q.sum(dim=0)
            ss_fp = (y_fp * y_fp).sum(dim=0)
            ss_q = (y_q * y_q).sum(dim=0)
            s_fpq = (y_fp * y_q).sum(dim=0)

        self.count += int(n)
        self.sum_fp += s_fp.detach().to("cpu").double()
        self.sum_q += s_q.detach().to("cpu").double()
        self.sumsq_fp += ss_fp.detach().to("cpu").double()
        self.sumsq_q += ss_q.detach().to("cpu").double()
        self.sum_fpq += s_fpq.detach().to("cpu").double()

    def finalize(
        self,
        alpha_min=0.7, alpha_max=1.5,
        beta_min=-0.5, beta_max=0.5,
        k_min=0.3, k_max=0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = max(self.count, 1)
        mean_fp = self.sum_fp / n
        mean_q = self.sum_q / n

        var_fp = (self.sumsq_fp / n) - (mean_fp * mean_fp)
        var_q = (self.sumsq_q / n) - (mean_q * mean_q)
        var_fp = var_fp.clamp_min(0.0) + self.eps
        var_q = var_q.clamp_min(0.0) + self.eps

        cov_qfp = (self.sum_fpq / n) - (mean_q * mean_fp)

        alpha = cov_qfp / var_q
        beta = mean_fp - alpha * mean_q
        alpha = alpha.clamp(alpha_min, alpha_max)
        beta = beta.clamp(beta_min, beta_max)

        # var(fp - q) = var_fp + var_q - 2*cov(fp,q)
        var_noise = var_fp + var_q - 2.0 * cov_qfp
        var_noise = var_noise.clamp_min(0.0) + self.eps

        snr = var_fp / var_noise
        K = snr / (1.0 + snr)
        K = K.clamp(k_min, k_max)

        return K.float(), alpha.float(), beta.float()


@torch.no_grad()
def fit_quantdir_for_single_layer_streaming(
    fp32_model: nn.Module,
    q_model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    layer_name: str,
    max_batches: int,
    mode: str,  # "conv" or "bn"
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    mods_fp = dict(fp32_model.named_modules())
    mods_q = dict(q_model.named_modules())
    if layer_name not in mods_fp or layer_name not in mods_q:
        return None

    m_fp = mods_fp[layer_name]
    m_q = mods_q[layer_name]

    if mode == "conv" and not isinstance(m_q, (nn.Conv2d, nn.Linear)):
        return None
    if mode == "bn" and not isinstance(m_q, nn.BatchNorm2d):
        return None

    holder = {"fp": None, "stats": None}

    def fp_hook(_m, _i, o):
        holder["fp"] = o.detach()

    def q_hook(_m, _i, o):
        y_fp = holder["fp"]
        if y_fp is None:
            return
        y_q = o.detach()
        if y_fp.shape != y_q.shape:
            return
        if holder["stats"] is None:
            C = int(y_fp.shape[1])
            holder["stats"] = _RunningStats(C=C)
        holder["stats"].update(y_fp, y_q)

    h1 = m_fp.register_forward_hook(fp_hook)
    h2 = m_q.register_forward_hook(q_hook)

    fp32_model.eval()
    q_model.eval()

    for bi, (x, _) in enumerate(calib_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        _ = fp32_model(x)
        _ = q_model(x)

    h1.remove()
    h2.remove()

    if holder["stats"] is None:
        return None
    return holder["stats"].finalize()


@torch.no_grad()
def apply_quantdir_post_quant(
    fp32_model: nn.Module,
    q_model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    conv_layers: List[str],
    bn_layers: List[str],
    max_batches_conv: int,
    max_batches_bn: int,
) -> Tuple[nn.Module, int, int]:
    """
    Post-quant insertion (your best-performing variant):
      enable_calibration -> calibrate
      enable_quantization
      fit+insert QuantDIR filters (offline)
    Returns:
      (q_model_with_quantdir, applied_conv_count, applied_bn_count)
    """
    applied_conv = 0
    applied_bn = 0

    # Conv/Linear layers: fit+insert one by one (memory safe)
    for lname in conv_layers:
        out = fit_quantdir_for_single_layer_streaming(
            fp32_model, q_model, calib_loader, device,
            layer_name=lname, max_batches=max_batches_conv, mode="conv"
        )
        if out is None:
            continue
        K, alpha, beta = out
        filt = QuantDIRFilter(alpha.numel(), K.to(device), alpha.to(device), beta.to(device))

        parent = q_model
        parts = lname.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], nn.Sequential(orig, filt))
        applied_conv += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # BN refresh (quant model) for selected BN layers
    mods_q = dict(q_model.named_modules())
    for lname in bn_layers:
        m = mods_q.get(lname, None)
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    for bi, (x, _) in enumerate(calib_loader):
        if bi >= max_batches_bn:
            break
        _ = q_model(x.to(device, non_blocking=True))

    for lname in bn_layers:
        m = mods_q.get(lname, None)
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # BN layers: fit+insert one by one
    for lname in bn_layers:
        out = fit_quantdir_for_single_layer_streaming(
            fp32_model, q_model, calib_loader, device,
            layer_name=lname, max_batches=max_batches_bn, mode="bn"
        )
        if out is None:
            continue
        K, alpha, beta = out
        filt = QuantDIRFilter(alpha.numel(), K.to(device), alpha.to(device), beta.to(device))

        parent = q_model
        parts = lname.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], nn.Sequential(orig, filt))
        applied_bn += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return q_model, applied_conv, applied_bn


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="QuantDIR ablation study: model + early/late/random selection.")

    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument("--resnet18", action="store_true", help="Use ResNet-18")
    mg.add_argument("--resnet34", action="store_true", help="Use ResNet-34")
    mg.add_argument("--resnet50", action="store_true", help="Use ResNet-50")
    mg.add_argument("--regnetx800mf", action="store_true", help="Use RegNetX-800MF")
    mg.add_argument("--mobilenet_v2", action="store_true", help="Use MobileNetV2")
    mg.add_argument("--vgg16", action="store_true", help="Use VGG16")
    mg.add_argument("--vgg19", action="store_true", help="Use VGG19")

    sg = p.add_mutually_exclusive_group(required=True)
    sg.add_argument("--early", action="store_true", help="Select only EARLY layers from CSV")
    sg.add_argument("--late", action="store_true", help="Select only LATE layers from CSV")
    sg.add_argument("--random", action="store_true", help="Select RANDOM layers from CSV")

    p.add_argument("--imagenet-root", type=str, default="../BackUp/test_dataset",
                   help="Path to ImageNet val folder (ImageFolder root).")
    p.add_argument("--csv-path", type=str, default=None,
                   help="Override CSV path (must have column 'layer'). If not set, uses your default mapping.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-calib", type=int, default=DEFAULT_NUM_CALIB)
    p.add_argument("--num-eval", type=int, default=DEFAULT_NUM_EVAL)
    p.add_argument("--calib-bs", type=int, default=DEFAULT_CALIB_BS)
    p.add_argument("--eval-bs", type=int, default=DEFAULT_EVAL_BS)
    p.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--n-select", type=int, default=DEFAULT_N_SELECT,
                   help="How many layers to sample from filtered pool (or all if fewer exist).")

    return p.parse_args()


def get_model_name_from_args(a) -> str:
    if a.resnet18:
        return "resnet18"
    if a.resnet34:
        return "resnet34"
    if a.resnet50:
        return "resnet50"
    if a.regnetx800mf:
        return "regnetx800mf"
    if a.mobilenet_v2:
        return "mobilenet_v2"
    if a.vgg16:
        return "vgg16"
    if a.vgg19:
        return "vgg19"
    raise ValueError("No model flag selected.")


def get_mode_from_args(a) -> str:
    if a.early:
        return "EARLY"
    if a.late:
        return "LATE"
    if a.random:
        return "RANDOM"
    raise ValueError("No mode flag selected.")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = get_model_name_from_args(args)
    mode = get_mode_from_args(args)

    # CSV path
    if args.csv_path is not None:
        csv_path = args.csv_path
    else:
        csv_map = default_csv_paths()
        if model_name not in csv_map:
            raise ValueError(f"No default CSV path for model '{model_name}'. Provide --csv-path.")
        csv_path = csv_map[model_name]

    # Loaders
    calib_loader, eval_loader = get_loaders(
        imagenet_root=args.imagenet_root,
        num_calib=args.num_calib,
        num_eval=args.num_eval,
        seed=args.seed,
        calib_bs=args.calib_bs,
        eval_bs=args.eval_bs,
        num_workers=args.workers,
    )

    # FP32 model
    fp32_model = build_fp32_model(model_name).to(device).eval()

    # INT4W8 baseline config
    extra_config = default_int4w8_extra_config()

    # ----------------------------
    # 1) FP32 metrics
    # ----------------------------
    fp32_top1, fp32_top5 = evaluate_topk(fp32_model, eval_loader, device)
    fp32_time = measure_inference_time_seconds(fp32_model, eval_loader, device)

    # ----------------------------
    # 2) INT4W8 baseline
    # ----------------------------
    q_base = build_fp32_model(model_name)
    q_base = prepare_by_platform(q_base, BackendType.Academic, extra_config).to(device)
    calibrate_and_quantize(q_base, calib_loader, device)

    base_top1, base_top5 = evaluate_topk(q_base, eval_loader, device)
    base_time = measure_inference_time_seconds(q_base, eval_loader, device)

    # ----------------------------
    # 3) QuantDIR (INT4W8 + QuantDIR filters)
    # ----------------------------
    q_quantdir = build_fp32_model(model_name)
    q_quantdir = prepare_by_platform(q_quantdir, BackendType.Academic, extra_config).to(device)
    calibrate_and_quantize(q_quantdir, calib_loader, device)

    sampled_layers = select_layers_from_csv(
        csv_path=csv_path,
        fp32_model=fp32_model,
        device=device,
        mode=mode,
        n_select=args.n_select,
        seed=args.seed,
    )
    buckets = bucket_layers_by_module_type(fp32_model, sampled_layers)

    t0 = time.perf_counter()
    q_quantdir, applied_conv, applied_bn = apply_quantdir_post_quant(
        fp32_model=fp32_model,
        q_model=q_quantdir,
        calib_loader=calib_loader,
        device=device,
        conv_layers=buckets["conv"],
        bn_layers=buckets["bn"],
        max_batches_conv=MAX_CALIB_BATCHES_CONV,
        max_batches_bn=MAX_CALIB_BATCHES_BN,
    )
    quantdir_offline_time = time.perf_counter() - t0

    qdir_top1, qdir_top5 = evaluate_topk(q_quantdir, eval_loader, device)
    qdir_time = measure_inference_time_seconds(q_quantdir, eval_loader, device)

    
    # Drop is typically reported vs FP32 and vs INT4W8 baseline.
    print(f"MODEL={model_name}  MODE={mode}  EvalN={len(eval_loader.dataset)}  CalibN={len(calib_loader.dataset)}")

    print(f"QuantDIR   : Top1={qdir_top1:.2f}  Top5={qdir_top5:.2f}  InferTime(s)={qdir_time:.2f}  "
          f"DropTop1(vsFP32)={qdir_top1 - fp32_top1:+.2f}  GainTop1(vsINT4W8)={qdir_top1 - base_top1:+.2f}  "
          f"OfflineQuantDIR(s)={quantdir_offline_time:.2f}")

    print(f"QuantDIRApplied: conv/linear={applied_conv}  bn={applied_bn}  (sampled={len(sampled_layers)})")


if __name__ == "__main__":
    # Multiprocessing-friendly on macOS / clusters
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

