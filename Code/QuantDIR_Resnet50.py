import io
import os
import time
import random

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization


# ----------------------------
# Config
# ----------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4           
WARMUP_STEPS = 5

NUM_CALIB = 1024
NUM_EVAL = 50000           

MAX_BATCHES_CONV = 40      
MAX_BATCHES_BN = 35       

IMAGENET_ROOT = "./test_dataset"


# ----------------------------
# Selected layers 
# ----------------------------
SELECTED_LAYERS = {
    "conv": [
        "conv1",
        "layer1.0.conv1",
        "layer1.0.conv2", "layer1.0.conv3", "layer1.0.downsample.0",
        "layer1.1.conv1",
        "layer2.0.conv1",
        "layer2.0.downsample.0",
        "layer3.0.conv3",
        "layer4.0.conv1", "layer4.0.conv2", "layer4.0.conv3", "layer4.0.downsample.0",
        "layer4.1.conv1","layer4.1.conv2","layer4.2.conv1","fc",
    ],
    "bn": [
         "layer1.0.bn3","layer1.1.bn3", "layer1.2.bn3",
        "layer2.1.bn1", "layer2.0.bn3",
        "layer2.1.bn3","layer2.2.bn1", "layer2.2.bn3","layer2.3.bn3",
         "layer3.0.bn1", "layer3.0.bn3","layer3.1.bn3", "layer3.2.bn3","layer3.3.bn3", "layer3.4.bn3","layer3.5.bn3",
          "layer4.0.bn1", 
         "layer4.0.bn3", "layer4.1.bn3", "layer4.2.bn3",
    ],
}


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data
# ----------------------------
def get_loaders(imagenet_root: str, num_calib: int = 1024, num_eval: int = 50000, seed: int = 42):
    set_seed(seed)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=imagenet_root, transform=transform)

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    calib_indices = indices[:num_calib]
    eval_indices = indices[num_calib:num_calib + num_eval]

    calib_subset = Subset(dataset, calib_indices)
    eval_subset = Subset(dataset, eval_indices)

    pin = torch.cuda.is_available()
    persistent = (NUM_WORKERS > 0)

    calib_loader = DataLoader(
        calib_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    return calib_loader, eval_loader


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def evaluate_topk(model: nn.Module, loader: DataLoader, device: torch.device, k: int = 5):
    model.eval()
    top1 = 0
    top5 = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        _, pred = out.topk(k, dim=1, largest=True, sorted=True)
        total += y.size(0)
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()
    return 100.0 * top1 / total, 100.0 * top5 / total


@torch.no_grad()
def measure_inference_blockwise(model: nn.Module, loader: DataLoader, device: torch.device, warmup_steps: int = 5):
    model.eval()

    # warmup
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

    # timed run (END-TO-END: loader + H2D + forward)
    t0 = time.time()
    total_images = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_images += x.size(0)

    total_runtime = time.time() - t0
    throughput = total_images / max(total_runtime, 1e-12)  # img/s
    latency_ms_batch = (total_runtime / max(len(loader), 1)) * 1000.0  # ms/batch
    return total_runtime, throughput, latency_ms_batch


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


# ----------------------------
# Mixed-precision config
# ----------------------------
def get_mixed_precision_config_w4a8_first_last_8bit():
    """
    Base: W4A8 for most layers, with conv1 and fc weights at 8-bit.
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
            "conv1": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": True, "pot_scale": False}},
            "layer3.0.downsample.0":{"w_qscheme":{"bit":8, "symmetry": False, "per_channel": True, "pot_scale": False}},
            "fc": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False}},
        },
    }
    return extra_config


# ----------------------------
# QuantDIR STREAMING stats 
# ----------------------------
@torch.no_grad()
def _reduce_per_channel_sums(o: torch.Tensor):
    """
    Returns (sum, sumsq, count) per channel for o.

    Conv/BN outputs: [N,C,H,W] -> reduce over N,H,W.
    Linear outputs:  [N,C]     -> reduce over N.
    """
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
def collect_streaming_stats(fp32_model: nn.Module,
                            q_model: nn.Module,
                            calib_loader: DataLoader,
                            device: torch.device,
                            target_layers: list,
                            max_batches: int = 40,
                            mode: str = "conv"):
    
    modules_fp32 = dict(fp32_model.named_modules())
    modules_q = dict(q_model.named_modules())

    acc = {}
    hooks = []

    def ensure_layer(name: str, C: int):
        if name not in acc:
            acc[name] = {
                "sum_q": torch.zeros(C, dtype=torch.float64),
                "sum_fp": torch.zeros(C, dtype=torch.float64),
                "sumsq_q": torch.zeros(C, dtype=torch.float64),
                "sumsq_fp": torch.zeros(C, dtype=torch.float64),
                "sum_qfp": torch.zeros(C, dtype=torch.float64),
                "count": 0,
            }

    # FP32 hook: clone to avoid in-place modification side effects
    def make_fp_hook(m_fp):
        def hook(_m, _i, o_fp):
            m_fp._latest_fp_out = o_fp.detach().clone()
        return hook

    def make_q_hook(layer_name: str, m_fp_ref):
        def hook(_m, _i, o_q):
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

            ensure_layer(layer_name, C)

            # Accumulate on CPU in float64
            acc[layer_name]["sum_q"] += s_q.detach().to("cpu").double()
            acc[layer_name]["sumsq_q"] += ss_q.detach().to("cpu").double()
            acc[layer_name]["sum_fp"] += s_fp.detach().to("cpu").double()
            acc[layer_name]["sumsq_fp"] += ss_fp.detach().to("cpu").double()
            acc[layer_name]["sum_qfp"] += s_qfp.detach().to("cpu").double()
            acc[layer_name]["count"] += int(n)
        return hook

    hooked_fp_modules = []
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
        hooked_fp_modules.append(m_fp)

    fp32_model.eval()
    q_model.eval()

    for bi, (x, _) in enumerate(calib_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)

        for m_fp in hooked_fp_modules:
            m_fp._latest_fp_out = None

        _ = fp32_model(x)
        _ = q_model(x)

    for h in hooks:
        h.remove()

    for m_fp in hooked_fp_modules:
        if hasattr(m_fp, "_latest_fp_out"):
            m_fp._latest_fp_out = None

    return acc


@torch.no_grad()
def QuantDIR_from_streaming(acc_entry: dict,
                          eps: float = 1e-6,
                          alpha_min: float = 0.7, alpha_max: float = 1.5,
                          beta_min: float = -0.5, beta_max: float = 0.5,
                          k_min: float = 0.3, k_max: float = 0.95):
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
    var_q = var_q.clamp_min(0.0) + eps
    var_fp = var_fp.clamp_min(0.0) + eps

    cov_qfp = (sum_qfp / n) - (mean_q * mean_fp)

    alpha = cov_qfp / var_q
    beta = mean_fp - alpha * mean_q
    alpha = alpha.clamp(alpha_min, alpha_max)
    beta = beta.clamp(beta_min, beta_max)

    var_noise = var_fp + var_q - 2.0 * cov_qfp
    var_noise = var_noise.clamp_min(0.0) + eps

    snr = var_fp / var_noise
    K = snr / (1.0 + snr)
    K = K.clamp(k_min, k_max)

    return K.float(), alpha.float(), beta.float()


class QuantDIRFilter(nn.Module):
    def __init__(self, num_channels: int, quantdir_gain: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.register_buffer("K", quantdir_gain.view(-1))
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


@torch.no_grad()
def apply_QuantDIR_to_layers_post_quant_streaming(fp32_model: nn.Module,
                                               q_model: nn.Module,
                                               calib_loader: DataLoader,
                                               device: torch.device,
                                               conv_layers: list,
                                               bn_layers: list,
                                               max_batches_conv: int = 40,
                                               max_batches_bn: int = 35,
                                               debug: bool = True):
    # Conv/Linear
    acc_conv = collect_streaming_stats(fp32_model, q_model, calib_loader, device,
                                       conv_layers, max_batches=max_batches_conv, mode="conv")

    if debug:
        print(f"[DEBUG] Conv stats collected for {len(acc_conv)}/{len(conv_layers)} layers")

    for lname in conv_layers:
        if lname not in acc_conv:
            continue
        K, alpha, beta = QuantDIR_from_streaming(acc_conv[lname])

        if debug:
            print(f"[DEBUG] {lname:<26} "
                  f"K[{K.min().item():.3f},{K.max().item():.3f}] "
                  f"a[{alpha.min().item():.3f},{alpha.max().item():.3f}] "
                  f"b[{beta.min().item():.3f},{beta.max().item():.3f}]")

        C = int(K.numel())
        QuantDIR = QuantDIRFilter(C, K.to(device), alpha.to(device), beta.to(device))

        parent = q_model
        parts = lname.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], nn.Sequential(orig, QuantDIR))

    # BN
    acc_bn = collect_streaming_stats(fp32_model, q_model, calib_loader, device,
                                     bn_layers, max_batches=max_batches_bn, mode="bn")

    if debug:
        print(f"[DEBUG] BN stats collected for {len(acc_bn)}/{len(bn_layers)} layers")

    for lname in bn_layers:
        if lname not in acc_bn:
            continue
        K, alpha, beta = QuantDIR_from_streaming(acc_bn[lname])

        if debug:
            print(f"[DEBUG] {lname:<26} "
                  f"K[{K.min().item():.3f},{K.max().item():.3f}] "
                  f"a[{alpha.min().item():.3f},{alpha.max().item():.3f}] "
                  f"b[{beta.min().item():.3f},{beta.max().item():.3f}]")

        C = int(K.numel())
        QuantDIR = QuantDIRFilter(C, K.to(device), alpha.to(device), beta.to(device))

        parent = q_model
        parts = lname.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], nn.Sequential(orig, QuantDIR))

    return q_model


# ----------------------------
# PTQ helper
# ----------------------------
@torch.no_grad()
def run_calibration(q_model: nn.Module, calib_loader: DataLoader, device: torch.device):
    q_model.eval()
    enable_calibration(q_model)
    for x, _ in calib_loader:
        _ = q_model(x.to(device, non_blocking=True))
    enable_quantization(q_model)


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type != "cuda":
        raise RuntimeError("This script is intended for CUDA GPU runs. (torch.cuda.is_available() is False)")

    torch.backends.cudnn.benchmark = True

    calib_loader, eval_loader = get_loaders(IMAGENET_ROOT, num_calib=NUM_CALIB, num_eval=NUM_EVAL, seed=SEED)
    print(f"Calib images: {len(calib_loader.dataset)} | Eval images: {len(eval_loader.dataset)} "
          f"| Batches(eval): {len(eval_loader)} | Batch size: {BATCH_SIZE} | workers: {NUM_WORKERS}")

    extra_config = get_mixed_precision_config_w4a8_first_last_8bit()

    # -------- FP32 --------
    fp32_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
   # fp32_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    fp32_top1, fp32_top5 = evaluate_topk(fp32_model, eval_loader, device)
    fp32_runtime, fp32_thr, fp32_lat_ms = measure_inference_blockwise(fp32_model, eval_loader, device, warmup_steps=WARMUP_STEPS)
    fp32_mem = get_runtime_mem_mb_params_buffers(fp32_model)
    fp32_ckpt = get_checkpoint_size_mb_state_dict(fp32_model)

    # -------- PTQ baseline --------
    #q_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    q_base =  models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
    q_base = prepare_by_platform(q_base, BackendType.Academic, extra_config).to(device)
    run_calibration(q_base, calib_loader, device)

    base_top1, base_top5 = evaluate_topk(q_base, eval_loader, device)
    base_runtime, base_thr, base_lat_ms = measure_inference_blockwise(q_base, eval_loader, device, warmup_steps=WARMUP_STEPS)
    base_mem = get_runtime_mem_mb_params_buffers(q_base)
    base_ckpt = get_checkpoint_size_mb_state_dict(q_base)

    # -------- PTQ + QuantDIR (one-time fit, then deploy) --------
    q_QuantDIR = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
    q_QuantDIR = prepare_by_platform(q_QuantDIR, BackendType.Academic, extra_config).to(device)
    run_calibration(q_QuantDIR, calib_loader, device)

    # OFFLINE one-time fitting time
    t0 = time.time()
    q_QuantDIR = apply_QuantDIR_to_layers_post_quant_streaming(
        fp32_model=fp32_model,
        q_model=q_QuantDIR,
        calib_loader=calib_loader,
        device=device,
        conv_layers=SELECTED_LAYERS["conv"],
        bn_layers=SELECTED_LAYERS["bn"],
        max_batches_conv=MAX_BATCHES_CONV,
        max_batches_bn=MAX_BATCHES_BN,
        debug=True,
    )
    QuantDIR_one_time_s = time.time() - t0

    # ONLINE inference metrics (no refit)
    kal_top1, kal_top5 = evaluate_topk(q_QuantDIR, eval_loader, device)
    kal_runtime, kal_thr, kal_lat_ms = measure_inference_blockwise(q_QuantDIR, eval_loader, device, warmup_steps=WARMUP_STEPS)
    kal_mem = get_runtime_mem_mb_params_buffers(q_QuantDIR)
    kal_ckpt = get_checkpoint_size_mb_state_dict(q_QuantDIR)


    # -------- Print summary --------
    print("\n" + "=" * 140)
    print("SUMMARY (QuantDIR one-time cost reported separately; inference metrics are END-TO-END ONLINE)")
    print("=" * 140)
    print(f"{'W4A8 + QuantDIR':<22} {kal_top1:>7.2f} {kal_top5:>7.2f} | {kal_runtime:>10.2f} {kal_thr:>18.1f} "
          f"{kal_lat_ms:>18.2f} | {kal_mem:>8.2f} {kal_ckpt:>9.2f}")
    print("-" * 140)
    print(f"One-time QuantDIR fit/insert cost (offline): {QuantDIR_one_time_s:.2f} s")
   
    print("=" * 140)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
