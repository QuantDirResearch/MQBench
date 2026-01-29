import io
import os
import time
import random
import gc
from typing import List, Set

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization


# ============================================================
# CONFIG
# ============================================================
SEED = 42
IMAGENET_ROOT = "./test_dataset"

NUM_CALIB = 1024
NUM_EVAL = 50000

BATCH_SIZE = 64

NUM_WORKERS_FP_BASE = 0     
NUM_WORKERS_QUANTDIR = 4     

PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

WARMUP_STEPS = 5

MAX_BATCHES_CONV = 40
MAX_BATCHES_BN = 35

# QuantDIR clamps
EPS = 1e-6
ALPHA_MIN, ALPHA_MAX = 0.25, 1.5
BETA_MIN,  BETA_MAX  = -0.25, 0.5
K_MIN,     K_MAX     = 0.25, 0.95


# ============================================================
# UNSTABLE CONV LAYERS
# ============================================================
SELECTED_CONV_LAYERS = [
    "stem.0",
    "trunk_output.block1.block1-0.f.b.0",
    "trunk_output.block1.block1-0.f.c.0",
    "trunk_output.block2.block2-0.f.a.0",
    "trunk_output.block2.block2-0.f.b.0",
    "trunk_output.block2.block2-0.f.c.0",
    "trunk_output.block2.block2-1.f.a.0",
    "trunk_output.block2.block2-2.f.a.0",
    "trunk_output.block3.block3-0.proj.0",
    "trunk_output.block3.block3-0.f.a.0",
    "trunk_output.block3.block3-0.f.c.0",
    "trunk_output.block3.block3-1.f.a.0",
    "trunk_output.block3.block3-2.f.a.0",
    "trunk_output.block3.block3-3.f.a.0",
    "trunk_output.block3.block3-3.f.c.0",
    "trunk_output.block3.block3-4.f.a.0",
    "trunk_output.block3.block3-5.f.a.0",
    "trunk_output.block3.block3-6.f.a.0",
    "trunk_output.block3.block3-6.f.b.0",
    "trunk_output.block4.block4-0.proj.0",
    "trunk_output.block4.block4-0.f.a.0",
    "trunk_output.block4.block4-0.f.c.0",
    "trunk_output.block4.block4-1.f.a.0",
    "trunk_output.block4.block4-1.f.c.0",
    "trunk_output.block4.block4-2.f.a.0",
    "trunk_output.block4.block4-3.f.c.0",
    "trunk_output.block4.block4-4.f.a.0",
    "trunk_output.block4.block4-4.f.c.0",
]


# ============================================================
# BN selection (derived from conv list)
# ============================================================
def derive_bn_from_conv_list(
    conv_layers: List[str],
    include_stem_bn: bool = True,
    include_a_bn: bool = True,
    include_c_bn: bool = True,
    include_proj_bn: bool = True,
) -> List[str]:
    """
    For each conv layer name ending with:
      - ".f.a.0" -> BN ".f.a.1"
      - ".f.c.0" -> BN ".f.c.1"
      - ".proj.0" -> BN ".proj.1"
    Also include stem BN "stem.1" if requested.

    This is architecture-consistent for torchvision RegNet.
    """
    bn: Set[str] = set()
    if include_stem_bn:
        bn.add("stem.1")

    for name in conv_layers:
        if include_a_bn and name.endswith(".f.a.0"):
            bn.add(name[:-1] + "1")  # ...f.a.0 -> ...f.a.1
        if include_c_bn and name.endswith(".f.c.0"):
            bn.add(name[:-1] + "1")  # ...f.c.0 -> ...f.c.1
        if include_proj_bn and name.endswith(".proj.0"):
            bn.add(name[:-1] + "1")  # ...proj.0 -> ...proj.1

    return sorted(bn)


SELECTED_BN_LAYERS = derive_bn_from_conv_list(
    SELECTED_CONV_LAYERS,
    include_stem_bn=True,
    include_a_bn=True,
    include_c_bn=True,
    include_proj_bn=True,
)


# ============================================================
# Repro
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================
def get_loaders(imagenet_root: str, num_calib: int, num_eval: int, seed: int, num_workers: int):
    set_seed(seed)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = datasets.ImageFolder(root=imagenet_root, transform=transform)

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    calib_indices = indices[:num_calib]
    eval_indices = indices[:num_eval]

    calib_subset = Subset(dataset, calib_indices)
    eval_subset = Subset(dataset, eval_indices)

    pin = torch.cuda.is_available()

    calib_loader = DataLoader(
        calib_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(PERSISTENT_WORKERS and num_workers > 0),
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(PERSISTENT_WORKERS and num_workers > 0),
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    )
    return calib_loader, eval_loader


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def evaluate_topk(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    top1 = top5 = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        _, pred = out.topk(5, dim=1, largest=True, sorted=True)
        total += y.size(0)
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()
    return 100.0 * top1 / total, 100.0 * top5 / total


@torch.no_grad()
def measure_runtime_fast(model: nn.Module, loader: DataLoader, device: torch.device, warmup_steps: int):
    """
    Accurate GPU runtime with CUDA events, avoids per-batch synchronize overhead.
    """
    model.eval()

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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        total_images = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            total_images += x.size(0)
        end.record()
        torch.cuda.synchronize()

        runtime_s = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.time()
        total_images = 0
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            total_images += x.size(0)
        runtime_s = time.time() - t0

    throughput = total_images / max(runtime_s, 1e-12)
    latency_ms_batch = (runtime_s / max(len(loader), 1)) * 1000.0
    return runtime_s, throughput, latency_ms_batch


# ============================================================
# MQBench config: W4A8 baseline + first/last 8-bit
# ============================================================
def get_mixed_precision_config_w4a8_first_last_8bit():
    """
    W4A8 everywhere by default, keep stem.0 and fc weights at 8-bit.
    """
    extra_config = {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMinMaxObserver",
            "w_fakequantize": "FixedFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {"bit": 4, "symmetry": False, "per_channel": True, "pot_scale": False},
            "a_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False},
        },
        "module_qconfig_dict": {
            "stem.0": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": True, "pot_scale": False}},
            "fc": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False}},
        },
    }
    return extra_config


@torch.no_grad()
def run_calibration(q_model: nn.Module, calib_loader: DataLoader, device: torch.device):
    q_model.eval()
    enable_calibration(q_model)
    for x, _ in calib_loader:
        _ = q_model(x.to(device, non_blocking=True))
    enable_quantization(q_model)
    q_model.eval()


# ============================================================
# QuantDIR streaming stats (self-contained)
# ============================================================
@torch.no_grad()
def _reduce_per_channel_sums(o: torch.Tensor):
    if o.dim() == 4:
        n = int(o.size(0) * o.size(2) * o.size(3))
        s = o.sum(dim=(0, 2, 3))
        ss = (o * o).sum(dim=(0, 2, 3))
        return s, ss, n
    if o.dim() == 2:
        n = int(o.size(0))
        s = o.sum(dim=0)
        ss = (o * o).sum(dim=0)
        return s, ss, n
    raise ValueError(f"Unsupported activation dim: {o.dim()}")


@torch.no_grad()
def collect_streaming_stats(
    fp32_model: nn.Module,
    q_model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    target_layers: list,
    max_batches: int,
    mode: str,
):
    modules_fp32 = dict(fp32_model.named_modules())
    modules_q = dict(q_model.named_modules())

    acc = {}
    hooks = []

    def ensure(name: str, C: int):
        if name not in acc:
            acc[name] = {
                "sum_q": torch.zeros(C, dtype=torch.float64),
                "sum_fp": torch.zeros(C, dtype=torch.float64),
                "sumsq_q": torch.zeros(C, dtype=torch.float64),
                "sumsq_fp": torch.zeros(C, dtype=torch.float64),
                "sum_qfp": torch.zeros(C, dtype=torch.float64),
                "count": 0,
            }

    def make_fp_hook(m_fp):
        def hook(_m, _i, o_fp):
            o_fp = o_fp[0] if isinstance(o_fp, (tuple, list)) else o_fp
            m_fp._latest_fp_out = o_fp.detach().clone()
        return hook

    def make_q_hook(layer_name: str, m_fp_ref):
        def hook(_m, _i, o_q):
            o_q = o_q[0] if isinstance(o_q, (tuple, list)) else o_q
            o_q = o_q.detach()
            o_fp = getattr(m_fp_ref, "_latest_fp_out", None)
            if o_fp is None or o_fp.shape != o_q.shape:
                return

            s_q, ss_q, n = _reduce_per_channel_sums(o_q)
            s_fp, ss_fp, _ = _reduce_per_channel_sums(o_fp)

            if o_q.dim() == 4:
                s_qfp = (o_q * o_fp).sum(dim=(0, 2, 3))
                C = int(o_q.size(1))
            else:
                s_qfp = (o_q * o_fp).sum(dim=0)
                C = int(o_q.size(1))

            ensure(layer_name, C)

            acc[layer_name]["sum_q"] += s_q.detach().cpu().double()
            acc[layer_name]["sumsq_q"] += ss_q.detach().cpu().double()
            acc[layer_name]["sum_fp"] += s_fp.detach().cpu().double()
            acc[layer_name]["sumsq_fp"] += ss_fp.detach().cpu().double()
            acc[layer_name]["sum_qfp"] += s_qfp.detach().cpu().double()
            acc[layer_name]["count"] += int(n)
        return hook

    hooked_fp = []
    for lname in target_layers:
        if lname not in modules_fp32 or lname not in modules_q:
            continue

        m_fp = modules_fp32[lname]
        m_q = modules_q[lname]

        if mode == "conv" and not isinstance(m_q, (nn.Conv2d, nn.Linear)):
            continue
        if mode == "bn" and not isinstance(m_q, nn.BatchNorm2d):
            continue

        hooks.append(m_fp.register_forward_hook(make_fp_hook(m_fp)))
        hooks.append(m_q.register_forward_hook(make_q_hook(lname, m_fp)))
        hooked_fp.append(m_fp)

    fp32_model.eval()
    q_model.eval()

    for bi, (x, _) in enumerate(calib_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)

        for m_fp in hooked_fp:
            m_fp._latest_fp_out = None

        _ = fp32_model(x)
        _ = q_model(x)

    for h in hooks:
        h.remove()

    for m_fp in hooked_fp:
        if hasattr(m_fp, "_latest_fp_out"):
            m_fp._latest_fp_out = None

    return acc


@torch.no_grad()
def QuantDIR_from_streaming(acc_entry: dict):
    n = max(int(acc_entry["count"]), 1)
    sum_q = acc_entry["sum_q"]
    sum_fp = acc_entry["sum_fp"]
    sumsq_q = acc_entry["sumsq_q"]
    sumsq_fp = acc_entry["sumsq_fp"]
    sum_qfp = acc_entry["sum_qfp"]

    mean_q = sum_q / n
    mean_fp = sum_fp / n

    var_q = (sumsq_q / n) - (mean_q * mean_q)
    var_fp = (sumsq_fp / n) - (mean_fp * mean_fp)
    var_q = var_q.clamp_min(0.0) + EPS
    var_fp = var_fp.clamp_min(0.0) + EPS

    cov = (sum_qfp / n) - (mean_q * mean_fp)

    alpha = (cov / var_q).clamp(ALPHA_MIN, ALPHA_MAX)
    beta = (mean_fp - alpha * mean_q).clamp(BETA_MIN, BETA_MAX)

    var_noise = (var_fp + var_q - 2.0 * cov).clamp_min(0.0) + EPS
    snr = var_fp / var_noise
    K = (snr / (1.0 + snr)).clamp(K_MIN, K_MAX)

    return K.float(), alpha.float(), beta.float()


class QuantDIRFilter(nn.Module):
    def __init__(self, K: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.register_buffer("K", K.view(-1))
        self.register_buffer("alpha", alpha.view(-1))
        self.register_buffer("beta", beta.view(-1))

    def forward(self, y_q: torch.Tensor) -> torch.Tensor:
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


def insert_after_module(model: nn.Module, layer_name: str, QuantDIR: nn.Module):
    parent = model
    parts = layer_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    orig = getattr(parent, parts[-1])
    setattr(parent, parts[-1], nn.Sequential(orig, QuantDIR))


@torch.no_grad()
def apply_QuantDIR_selected(
    fp32_model: nn.Module,
    q_model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    conv_layers: List[str],
    bn_layers: List[str],
    debug: bool = True,
):
    # Conv/Linear
    acc_conv = collect_streaming_stats(
        fp32_model, q_model, calib_loader, device,
        conv_layers, max_batches=MAX_BATCHES_CONV, mode="conv"
    )

    if debug:
        print(f"[DEBUG] Conv stats collected for {len(acc_conv)}/{len(conv_layers)} layers")
        missing = [n for n in conv_layers if n not in acc_conv]
        if missing:
            print("[DEBUG] Missing conv layers (first 30):", missing[:30])

    for lname in conv_layers:
        if lname not in acc_conv:
            continue
        K, a, b = QuantDIR_from_streaming(acc_conv[lname])
        if debug:
            print(
                f"[QuantDIR-CONV] {lname:<45} "
                f"K[{K.min():.3f},{K.max():.3f}] "
                f"a[{a.min():.3f},{a.max():.3f}] "
                f"b[{b.min():.3f},{b.max():.3f}]"
            )
        insert_after_module(q_model, lname, QuantDIRFilter(K.to(device), a.to(device), b.to(device)))

    # BN
    acc_bn = collect_streaming_stats(
        fp32_model, q_model, calib_loader, device,
        bn_layers, max_batches=MAX_BATCHES_BN, mode="bn"
    )

    if debug:
        print(f"[DEBUG] BN stats collected for {len(acc_bn)}/{len(bn_layers)} layers")
        missing_bn = [n for n in bn_layers if n not in acc_bn]
        if missing_bn:
            print("[DEBUG] Missing BN layers:", missing_bn)

    for lname in bn_layers:
        if lname not in acc_bn:
            continue
        K, a, b = QuantDIR_from_streaming(acc_bn[lname])
        if debug:
            print(
                f"[QuantDIR-BN ] {lname:<45} "
                f"K[{K.min():.3f},{K.max():.3f}] "
                f"a[{a.min():.3f},{a.max():.3f}] "
                f"b[{b.min():.3f},{b.max():.3f}]"
            )
        insert_after_module(q_model, lname, QuantDIRFilter(K.to(device), a.to(device), b.to(device)))

    return q_model


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print(f"[SELECT] conv layers: {len(SELECTED_CONV_LAYERS)}")
    print(f"[SELECT] bn layers  : {len(SELECTED_BN_LAYERS)}")
    print("[SELECT] BN list (derived):")
    for n in SELECTED_BN_LAYERS:
        print("   ", n)

    # FP32 + baseline loaders (workers = NUM_WORKERS_FP_BASE)
    calib_loader, eval_loader = get_loaders(
        IMAGENET_ROOT, NUM_CALIB, NUM_EVAL, SEED, num_workers=NUM_WORKERS_FP_BASE
    )

    # QuantDIR loaders (workers = NUM_WORKERS_QUANTDIR)
    calib_loader_qdir, eval_loader_qdir = get_loaders(
        IMAGENET_ROOT, NUM_CALIB, NUM_EVAL, SEED, num_workers=NUM_WORKERS_QUANTDIR
    )

    print(
        f"Calib images: {len(calib_loader.dataset)} | Eval images: {len(eval_loader.dataset)} | "
        f"Batches(eval): {len(eval_loader)} | Batch size: {BATCH_SIZE}"
    )
    print(
        f"[Workers] FP32+Base num_workers={NUM_WORKERS_FP_BASE} | QuantDIR num_workers={NUM_WORKERS_QUANTDIR}"
    )

    extra_config = get_mixed_precision_config_w4a8_first_last_8bit()

    # -------- FP32 --------
    fp32_model = models.regnet_x_800mf(
        weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1
    ).to(device).eval()

    fp32_top1, fp32_top5 = evaluate_topk(fp32_model, eval_loader, device)
    fp32_runtime, fp32_thr, fp32_lat = measure_runtime_fast(
        fp32_model, eval_loader, device, warmup_steps=WARMUP_STEPS
    )

    # -------- W4A8 baseline --------
    q_base = models.regnet_x_800mf(
        weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1
    )
    q_base = prepare_by_platform(q_base, BackendType.Academic, extra_config).to(device)
    run_calibration(q_base, calib_loader, device)

    base_top1, base_top5 = evaluate_topk(q_base, eval_loader, device)
    base_runtime, base_thr, base_lat = measure_runtime_fast(
        q_base, eval_loader, device, warmup_steps=WARMUP_STEPS
    )

    # -------- W4A8 + QuantDIR --------
    q_dir = models.regnet_x_800mf(
        weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1
    )
    q_dir = prepare_by_platform(q_dir, BackendType.Academic, extra_config).to(device)
    run_calibration(q_dir, calib_loader_qdir, device)

    t0 = time.time()
    q_dir = apply_QuantDIR_selected(
        fp32_model=fp32_model,
        q_model=q_dir,
        calib_loader=calib_loader_qdir,
        device=device,
        conv_layers=SELECTED_CONV_LAYERS,
        bn_layers=SELECTED_BN_LAYERS,
        debug=True,
    )
    qdir_fit_time = time.time() - t0

    qdir_top1, qdir_top5 = evaluate_topk(q_dir, eval_loader_qdir, device)
    qdir_runtime, qdir_thr, qdir_lat = measure_runtime_fast(
        q_dir, eval_loader_qdir, device, warmup_steps=WARMUP_STEPS
    )

    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(f"{'Method':<14} {'Top-1':>7} {'Top-5':>7} | {'Runtime(s)':>10} {'Thrp(img/s)':>14} {'Lat(ms/b)':>12}")
    print("-" * 130)
    print(f"{'FP32':<14} {fp32_top1:>7.2f} {fp32_top5:>7.2f} | {fp32_runtime:>10.2f} {fp32_thr:>14.1f} {fp32_lat:>12.2f}")
    print(f"{'W4A8 Base':<14} {base_top1:>7.2f} {base_top5:>7.2f} | {base_runtime:>10.2f} {base_thr:>14.1f} {base_lat:>12.2f}")
    print(f"{'W4A8+QuantDIR':<14} {qdir_top1:>7.2f} {qdir_top5:>7.2f} | {qdir_runtime:>10.2f} {qdir_thr:>14.1f} {qdir_lat:>12.2f}")
    print("-" * 130)
    print(f"One-time QuantDIR repair cost (offline): {qdir_fit_time:.2f} s")
    print(f"Gain vs baseline (Top-1): {qdir_top1 - base_top1:+.2f} pts")
    print("=" * 130)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()