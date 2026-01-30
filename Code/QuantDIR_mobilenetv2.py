

import os
import io
import time
import random
import gc
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization


# =========================
# 0) CONFIG
# =========================
SEED = 42
IMAGENET_ROOT = "./test_dataset"   # ImageFolder root (must have class subfolders)
NUM_CALIB = 1024
NUM_EVAL  = 50000
CALIB_BS  = 1
EVAL_BS   = 64
NUM_WORKERS = 2

MODEL_NAME = "mobilenet_v2"  

SELECTED_CONV_LINEAR = [
    # MobileNetV2 examples:
    "features.0.0",
    "features.1.conv.0.0",
    "features.1.conv.1",
    "features.2.conv.0.0",
    "features.2.conv.1.0",
    "features.2.conv.2",
    "features.3.conv.1.0",
    "features.4.conv.2",
    "features.5.conv.1.0",
    "features.7.conv.2",
    "features.10.conv.0.0",
    "features.10.conv.2",
    "features.11.conv.1.0",
    "features.11.conv.2",
    "features.12.conv.0.0",
    "features.12.conv.1.0",
    "features.13.conv.1.0",
    "features.13.conv.2",
    "features.15.conv.1.0",
    "features.15.conv.2",
    "features.16.conv.1.0",
    "features.17.conv.0.0",
    "features.17.conv.2",
    "features.18.0",
    "classifier.1",
]


SELECTED_BN = [
    "features.1.conv.2",
    "features.2.conv.3",
    "features.3.conv.3",
    "features.6.conv.3",
    "features.7.conv.3",
    "features.17.conv.3",
]

# Fit budget
MAX_CALIB_BATCHES_PER_LAYER = 80

# QuantDIR parameter clamps
ALPHA_MIN, ALPHA_MAX = 0.85, 1.15
BETA_MIN,  BETA_MAX  = -0.10, 0.10
K_MIN,     K_MAX     = 0.30, 0.95

EPS = 1e-6
WARMUP_STEPS = 5  # used for stable throughput timing


# =========================
# 1) Repro + data
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_loaders():
    set_seed(SEED)
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root=IMAGENET_ROOT, transform=tfm)

    idx = list(range(len(ds)))
    random.shuffle(idx)

    calib_idx = idx[:NUM_CALIB]
    eval_idx  = idx[NUM_CALIB:NUM_CALIB + NUM_EVAL]

    calib_ds = Subset(ds, calib_idx)
    eval_ds  = Subset(ds, eval_idx)

    pin = torch.cuda.is_available()
    calib_loader = DataLoader(
        calib_ds, batch_size=CALIB_BS, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=EVAL_BS, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    return calib_loader, eval_loader


@torch.no_grad()
def evaluate_top1_top5(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    top1 = top5 = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        _, pred5 = out.topk(5, dim=1, largest=True, sorted=True)
        total += y.size(0)
        top1 += (pred5[:, 0] == y).sum().item()
        top5 += (pred5 == y.unsqueeze(1)).any(dim=1).sum().item()
    return 100.0 * top1 / total, 100.0 * top5 / total


@torch.no_grad()
def measure_runtime_throughput(model: nn.Module, loader: DataLoader, device: torch.device, warmup_steps: int = 5):
    model.eval()

    # Warmup
    it = iter(loader)
    for _ in range(warmup_steps):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed run
    t0 = time.time()
    total_images = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_images += x.size(0)

    runtime_s = time.time() - t0
    throughput = total_images / max(runtime_s, 1e-12)  # img/s
    return runtime_s, throughput


def get_runtime_mem_mb_params_buffers(model: nn.Module) -> float:
    nbytes = 0
    for p in model.parameters():
        nbytes += p.numel() * p.element_size()
    for b in model.buffers():
        nbytes += b.numel() * b.element_size()
    return nbytes / (1024 ** 2)


def get_checkpoint_size_mb_state_dict(model: nn.Module) -> float:
    bio = io.BytesIO()
    torch.save(model.state_dict(), bio)
    return bio.getbuffer().nbytes / (1024 ** 2)


# =========================
# 2) Model builders
# =========================
def build_fp32():
    if MODEL_NAME == "mobilenet_v2":
        return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if MODEL_NAME == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    raise ValueError("MODEL_NAME must be mobilenet_v2 or resnet50")


def build_int4_from_fp32(fp32_model: nn.Module, device: torch.device):
    extra_config = {
        "extra_qconfig_dict": {
            "w_observer": "MSEObserver",
            "a_observer": "EMAMSEObserver",
            "w_fakequantize": "FixedFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {"bit": 4, "symmetry": False, "per_channel": True, "pot_scale": False},
            "a_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False},
        },
        # Example mixed precision (keep as you had)
        "module_qconfig_dict": {
            "features.0.0": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": True, "pot_scale": False}},
            "classifier.1": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False}},
        },
    }
    q_model = prepare_by_platform(fp32_model, BackendType.Academic, extra_config).to(device)
    return q_model


@torch.no_grad()
def calibrate(q_model: nn.Module, calib_loader: DataLoader, device: torch.device):
    enable_calibration(q_model)
    q_model.eval()
    for x, _ in calib_loader:
        x = x.to(device, non_blocking=True)
        _ = q_model(x)
    enable_quantization(q_model)
    q_model.eval()


# =========================
# 3) QuantDIR components (streaming stats)
# =========================
class QuantDIRLayer(nn.Module):
  
    def __init__(self, C: int, K: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.register_buffer("K",     K.view(-1))
        self.register_buffer("alpha", alpha.view(-1))
        self.register_buffer("beta",  beta.view(-1))

    def forward(self, y_q):
        if y_q.dim() == 4:
            K = self.K.view(1, -1, 1, 1)
            a = self.alpha.view(1, -1, 1, 1)
            b = self.beta.view(1, -1, 1, 1)
        else:
            K = self.K.view(1, -1)
            a = self.alpha.view(1, -1)
            b = self.beta.view(1, -1)

        corrected = a * y_q + b
        return K * y_q + (1.0 - K) * corrected


class RunningStats:
    def __init__(self, C: int):
        self.count = 0
        self.sum_fp   = torch.zeros(C, dtype=torch.float64)
        self.sum_q    = torch.zeros(C, dtype=torch.float64)
        self.sumsq_fp = torch.zeros(C, dtype=torch.float64)
        self.sumsq_q  = torch.zeros(C, dtype=torch.float64)
        self.sum_fpq  = torch.zeros(C, dtype=torch.float64)

    def update(self, y_fp, y_q):
        # y: [N,C,H,W] or [N,C]
        if y_fp.dim() == 4:
            n = y_fp.size(0) * y_fp.size(2) * y_fp.size(3)
            s_fp = y_fp.sum(dim=(0,2,3))
            s_q  = y_q.sum(dim=(0,2,3))
            ss_fp = (y_fp*y_fp).sum(dim=(0,2,3))
            ss_q  = (y_q*y_q).sum(dim=(0,2,3))
            s_fpq = (y_fp*y_q).sum(dim=(0,2,3))
        else:
            n = y_fp.size(0)
            s_fp = y_fp.sum(dim=0)
            s_q  = y_q.sum(dim=0)
            ss_fp = (y_fp*y_fp).sum(dim=0)
            ss_q  = (y_q*y_q).sum(dim=0)
            s_fpq = (y_fp*y_q).sum(dim=0)

        self.count += int(n)
        self.sum_fp   += s_fp.detach().cpu().double()
        self.sum_q    += s_q.detach().cpu().double()
        self.sumsq_fp += ss_fp.detach().cpu().double()
        self.sumsq_q  += ss_q.detach().cpu().double()
        self.sum_fpq  += s_fpq.detach().cpu().double()

    def finalize(self):
        n = max(self.count, 1)
        mean_fp = self.sum_fp / n
        mean_q  = self.sum_q / n

        var_fp = (self.sumsq_fp / n) - mean_fp*mean_fp
        var_q  = (self.sumsq_q  / n) - mean_q*mean_q
        var_fp = var_fp.clamp_min(0.0) + EPS
        var_q  = var_q.clamp_min(0.0) + EPS

        cov = (self.sum_fpq / n) - mean_fp*mean_q

        alpha = (cov / var_q).clamp(ALPHA_MIN, ALPHA_MAX)
        beta  = (mean_fp - alpha*mean_q).clamp(BETA_MIN, BETA_MAX)

       
        var_noise = (var_fp + var_q - 2.0*cov).clamp_min(0.0) + EPS
        snr = var_fp / var_noise
        K = (snr / (1.0 + snr)).clamp(K_MIN, K_MAX)

        return K.float(), alpha.float(), beta.float()


@torch.no_grad()
def fit_one_layer(fp32_model, int4_model, calib_loader, device, layer_name: str, max_batches: int):
    fp_mods = dict(fp32_model.named_modules())
    q_mods  = dict(int4_model.named_modules())
    if layer_name not in fp_mods or layer_name not in q_mods:
        return None

    m_fp = fp_mods[layer_name]
    m_q  = q_mods[layer_name]

    if not isinstance(m_q, (nn.Conv2d, nn.Linear)):
        return None

    holder = {"fp": None, "stats": None, "seen": 0}

    def fp_hook(_m, _i, o):
        holder["fp"] = (o[0] if isinstance(o, (tuple, list)) else o).detach()

    def q_hook(_m, _i, o):
        y_q = (o[0] if isinstance(o, (tuple, list)) else o).detach()
        y_fp = holder["fp"]
        if y_fp is None or y_fp.shape != y_q.shape:
            return
        if holder["stats"] is None:
            C = int(y_fp.shape[1]) if y_fp.dim() >= 2 else None
            if C is None:
                return
            holder["stats"] = RunningStats(C)
        holder["stats"].update(y_fp, y_q)
        holder["seen"] += 1

    h1 = m_fp.register_forward_hook(fp_hook)
    h2 = m_q.register_forward_hook(q_hook)

    fp32_model.eval()
    int4_model.eval()

    for bi, (x, _) in enumerate(calib_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        _ = fp32_model(x)
        _ = int4_model(x)

    h1.remove()
    h2.remove()

    if holder["stats"] is None or holder["seen"] == 0:
        return None
    return holder["stats"].finalize()


def insert_after_module(model: nn.Module, layer_name: str, QuantDIR: nn.Module):
    parent = model
    parts = layer_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    orig = getattr(parent, parts[-1])
    setattr(parent, parts[-1], nn.Sequential(orig, QuantDIR))


@torch.no_grad()
def bn_refresh(model, calib_loader, device, bn_names):
    if not bn_names:
        return
    mods = dict(model.named_modules())
    for n in bn_names:
        m = mods.get(n, None)
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    for x, _ in calib_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
    for n in bn_names:
        m = mods.get(n, None)
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


@torch.no_grad()
def apply_QuantDIR_selected(fp32_model, int4_model, calib_loader, device, selected_layers):
    applied = []
    for lname in selected_layers:
        out = fit_one_layer(fp32_model, int4_model, calib_loader, device, lname, MAX_CALIB_BATCHES_PER_LAYER)
        if out is None:
            print(f"[SKIP] {lname} (not found / not Conv/Linear / no stats)")
            continue
        K, a, b = out
        quantdir = QuantDIRLayer(a.numel(), K.to(device), a.to(device), b.to(device))
        insert_after_module(int4_model, lname, quantdir)
        applied.append(lname)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[QuantDIR] Applied to {len(applied)} layers.")
    if applied:
        print("         First few:", applied[:10])
    return int4_model, applied


# =========================
# 4) Main
# =========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("ImageNet root:", IMAGENET_ROOT)
    print("Model:", MODEL_NAME)

    calib_loader, eval_loader = get_loaders()

    # -------------------------
    # FP32 reference
    # -------------------------
    fp32 = build_fp32().to(device).eval()
    fp32_top1, fp32_top5 = evaluate_top1_top5(fp32, eval_loader, device)
    fp32_runtime, fp32_thr = measure_runtime_throughput(fp32, eval_loader, device, warmup_steps=WARMUP_STEPS)
    fp32_mem  = get_runtime_mem_mb_params_buffers(fp32)
    fp32_ckpt = get_checkpoint_size_mb_state_dict(fp32)

    print(f"[FP32] Top-1: {fp32_top1:.2f}% | Top-5: {fp32_top5:.2f}%")

    # -------------------------
    # INT4 baseline
    # -------------------------
    q_base = build_int4_from_fp32(build_fp32().eval(), device)
    calibrate(q_base, calib_loader, device)
    base_top1, base_top5 = evaluate_top1_top5(q_base, eval_loader, device)
    base_runtime, base_thr = measure_runtime_throughput(q_base, eval_loader, device, warmup_steps=WARMUP_STEPS)
    # base_mem  = get_runtime_mem_mb_params_buffers(q_base)
    # base_ckpt = get_checkpoint_size_mb_state_dict(q_base)

    print(f"[INT4 Baseline] Top-1: {base_top1:.2f}% | Top-5: {base_top5:.2f}%")

    # -------------------------
    # INT4 + QuantDIR(selected)
    # -------------------------
    q_quantDIR = build_int4_from_fp32(build_fp32().eval(), device)
    calibrate(q_quantDIR, calib_loader, device)

    # Optional BN refresh before fitting corrections
    bn_refresh(q_quantDIR, calib_loader, device, SELECTED_BN)
    t0 = time.time()
    q_quantDIR, applied_layers = apply_QuantDIR_selected(fp32, q_quantDIR, calib_loader, device, SELECTED_CONV_LINEAR)
    QuantDIR_fit_time = time.time() - t0

    kal_top1, kal_top5 = evaluate_top1_top5(q_quantDIR, eval_loader, device)
    kal_runtime, kal_thr = measure_runtime_throughput(q_quantDIR, eval_loader, device, warmup_steps=WARMUP_STEPS)
    # kal_mem  = get_runtime_mem_mb_params_buffers(q_quantDIR)
    # kal_ckpt = get_checkpoint_size_mb_state_dict(q_quantDIR)

    print(f"[INT4 + QuantDIR(selected)] Top-1: {kal_top1:.2f}% | Top-5: {kal_top5:.2f}% | "
          f"Gain vs base: {kal_top1 - base_top1:+.2f} (Top-1)")

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 118)
    print("SUMMARY (Latency omitted)")
    print("=" * 118)
    print(f"{'INT4+QuantDIR':<18} {kal_top1:>7.2f} {kal_top5:>7.2f} | {kal_runtime:>10.2f} {kal_thr:>18.1f} ")
    print("-" * 118)
    print(f"Accuracy recovery: {kal_top1 - base_top1:+.2f} (Top-1) | {kal_top5 - base_top5:+.2f} (Top-5)")
    print(f"QuantDIR fit+insert time (offline): {QuantDIR_fit_time:.2f}s")
    print(f"QuantDIR filters inserted: {len(applied_layers)}")
    print("=" * 118)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
