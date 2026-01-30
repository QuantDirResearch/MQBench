import io
import os
import time
import random
from collections import defaultdict

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
WARMUP_STEPS = 5

NUM_CALIB = 1024
NUM_EVAL = 50000

# Dataloader workers per method (YOUR REQUEST)
WORKERS_FP32_INT4 = 0
WORKERS_QUANTDIR  = 4

MAX_BATCHES_CONV = 40
MAX_BATCHES_BN = 35

SELECTED_LAYERS = {
    "conv": [
        "conv1",
        "layer1.0.conv1", "layer1.1.conv1",
        "layer2.0.conv1", "layer2.0.conv2", "layer2.0.downsample.0",
        "layer4.0.conv1", "layer4.0.conv2", "layer4.0.downsample.0", "layer4.1.conv2",
    ],
    "bn": [
        "layer1.0.bn2", "layer1.1.bn2",
        "layer2.0.bn2", "layer2.1.bn2",
        "layer3.0.bn2", "layer3.1.bn2",
        "layer4.0.bn2", "layer4.1.bn2",
    ],
}


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data
# ----------------------------
def _build_imagenet_dataset(imagenet_root):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(root=imagenet_root, transform=transform)


def _make_indices(n, num_calib, num_eval, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    calib_idx = idx[:num_calib]
    eval_idx = idx[:num_eval] 
    return calib_idx, eval_idx


def get_loaders_with_indices(dataset, calib_indices, eval_indices, batch_size, num_workers):
    calib_subset = Subset(dataset, calib_indices)
    eval_subset = Subset(dataset, eval_indices)

    calib_loader = DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return calib_loader, eval_loader


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def evaluate_topk(model, loader, device, k=5):
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
def measure_inference_blockwise(model, loader, device, warmup_steps=5):
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

    t0 = time.time()
    total_images = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_images += x.size(0)

    total_runtime = time.time() - t0
    throughput = total_images / max(total_runtime, 1e-12)
    latency_ms_batch = (total_runtime / max(len(loader), 1)) * 1000.0
    return total_runtime, throughput, latency_ms_batch


# ----------------------------
# Size helpers (same as before)
# ----------------------------
def theoretical_w4a8_model_size_mb(model):
    total_bits = 0
    activation_layers = 0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total_bits += m.weight.numel() * 4

            out_ch = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features

            if m.bias is not None:
                total_bits += out_ch * 32

            total_bits += out_ch * (32 + 32)
            activation_layers += 1

    total_bits += activation_layers * (32 + 32)
    return total_bits / 8 / (1024 ** 2)


def quantdir_overhead_mb(model: nn.Module) -> float:
    nbytes = 0
    for m in model.modules():
        if hasattr(m, "K") and hasattr(m, "alpha") and hasattr(m, "beta"):
            for t in (m.K, m.alpha, m.beta):
                if isinstance(t, torch.Tensor):
                    nbytes += t.numel() * t.element_size()
    return nbytes / (1024 ** 2)


def theoretical_w4a8_plus_quantdir_mb(model: nn.Module) -> float:
    return theoretical_w4a8_model_size_mb(model) + quantdir_overhead_mb(model)


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


def get_theoretical_size_mb(model, weight_bits=32):
    param_count = sum(p.numel() for p in model.parameters())
    return (param_count * weight_bits) / 8 / (1024 ** 2)


# ----------------------------
# MQBench config (same)
# ----------------------------
def get_mixed_precision_config_w4a8_first_last_8bit():
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
            "layer3.0.downsample.0": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": True, "pot_scale": False}},
            "fc": {"w_qscheme": {"bit": 8, "symmetry": False, "per_channel": False, "pot_scale": False}},
        },
    }
    return extra_config


# ----------------------------
# QuantDIR logic (UNCHANGED)
# ----------------------------
@torch.no_grad()
def compute_QuantDIR_params_batched(fp32_acts, quant_acts, eps=1e-6,
                                   alpha_min=0.7, alpha_max=1.5,
                                   beta_min=-0.5, beta_max=0.5,
                                   k_min=0.3, k_max=0.95):
    if fp32_acts.dim() == 4:
        y_fp = fp32_acts.permute(1, 0, 2, 3).reshape(fp32_acts.size(1), -1)
        y_q = quant_acts.permute(1, 0, 2, 3).reshape(quant_acts.size(1), -1)
    else:
        y_fp = fp32_acts.t()
        y_q = quant_acts.t()

    mean_fp = y_fp.mean(dim=1)
    mean_q = y_q.mean(dim=1)

    var_q = y_q.var(dim=1, unbiased=False) + eps
    var_fp = y_fp.var(dim=1, unbiased=False) + eps

    y_q_centered = y_q - mean_q.unsqueeze(1)
    y_fp_centered = y_fp - mean_fp.unsqueeze(1)
    cov_q_fp = (y_q_centered * y_fp_centered).mean(dim=1)

    alpha = cov_q_fp / var_q
    beta = mean_fp - alpha * mean_q

    alpha = alpha.clamp(alpha_min, alpha_max)
    beta = beta.clamp(beta_min, beta_max)

    noise = y_fp - y_q
    var_noise = noise.var(dim=1, unbiased=False) + eps

    snr = var_fp / var_noise
    K = snr / (1.0 + snr)
    K = K.clamp(k_min, k_max)
    return K, alpha, beta


@torch.no_grad()
def collect_activations(fp32_model, q_model, calib_loader, device, target_layers,
                        max_batches=40, mode="conv"):
    modules_fp32 = dict(fp32_model.named_modules())
    modules_q = dict(q_model.named_modules())

    fp32_feats = defaultdict(list)
    q_feats = defaultdict(list)
    hooks = []

    for lname in target_layers:
        if lname not in modules_fp32 or lname not in modules_q:
            continue

        m_fp = modules_fp32[lname]
        m_q = modules_q[lname]

        if mode == "conv" and not isinstance(m_q, (nn.Conv2d, nn.Linear)):
            continue
        if mode == "bn" and not isinstance(m_q, nn.BatchNorm2d):
            continue

        def make_fp_hook(name):
            def hook(m, i, o):
                fp32_feats[name].append(o.detach().cpu())
            return hook

        def make_q_hook(name):
            def hook(m, i, o):
                q_feats[name].append(o.detach().cpu())
            return hook

        hooks.append(m_fp.register_forward_hook(make_fp_hook(lname)))
        hooks.append(m_q.register_forward_hook(make_q_hook(lname)))

    for bi, (x, _) in enumerate(calib_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        _ = fp32_model(x)
        _ = q_model(x)

    for h in hooks:
        h.remove()

    out = {}
    for name in target_layers:
        if len(fp32_feats[name]) == 0 or len(q_feats[name]) == 0:
            continue
        fp32_cat = torch.cat(fp32_feats[name], dim=0)
        q_cat = torch.cat(q_feats[name], dim=0)
        if fp32_cat.shape != q_cat.shape:
            continue
        out[name] = (fp32_cat, q_cat)

    return out


class QuantDIRFilter(nn.Module):
    def __init__(self, num_channels, QuantDIR_gain, alpha, beta):
        super().__init__()
        self.register_buffer("K", QuantDIR_gain.view(-1))
        self.register_buffer("alpha", alpha.view(-1))
        self.register_buffer("beta", beta.view(-1))

    def forward(self, y_q):
        if y_q.dim() == 4:
            K = self.K.view(1, -1, 1, 1)
            alpha = self.alpha.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        else:
            K = self.K.view(1, -1)
            alpha = self.alpha.view(1, -1)
            beta = self.beta.view(1, -1)

        corrected = alpha * y_q + beta
        estimate = K * y_q + (1 - K) * corrected
        return estimate


@torch.no_grad()
def apply_QuantDIR_to_layers_post_quant(fp32_model, q_model, calib_loader, device,
                                       conv_layers, bn_layers,
                                       max_batches_conv=40, max_batches_bn=35):
    act_dict = collect_activations(fp32_model, q_model, calib_loader, device,
                                   conv_layers, max_batches=max_batches_conv, mode="conv")
    for lname, (y_fp, y_q) in act_dict.items():
        K, alpha, beta = compute_QuantDIR_params_batched(y_fp, y_q)
        C = y_fp.shape[1]
        qd = QuantDIRFilter(C, K.to(device), alpha.to(device), beta.to(device))

        parent = q_model
        parts = lname.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], nn.Sequential(orig, qd))

    act_dict = collect_activations(fp32_model, q_model, calib_loader, device,
                                   bn_layers, max_batches=max_batches_bn, mode="bn")
    for lname, (y_fp, y_q) in act_dict.items():
        K, alpha, beta = compute_QuantDIR_params_batched(y_fp, y_q)
        C = y_fp.shape[1]
        qd = QuantDIRFilter(C, K.to(device), alpha.to(device), beta.to(device))

        parent = q_model
        parts = lname.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, parts[-1])
        setattr(parent, parts[-1], nn.Sequential(orig, qd))

    return q_model


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    imagenet_root = ".test_dataset"

    # Build dataset ONCE, indices ONCE (so all methods use identical examples)
    dataset = _build_imagenet_dataset(imagenet_root)
    calib_idx, eval_idx = _make_indices(len(dataset), NUM_CALIB, NUM_EVAL, SEED)

    # Two loader variants per your request
    calib0, eval0 = get_loaders_with_indices(dataset, calib_idx, eval_idx, BATCH_SIZE, WORKERS_FP32_INT4)
    calib4, eval4 = get_loaders_with_indices(dataset, calib_idx, eval_idx, BATCH_SIZE, WORKERS_QUANTDIR)

    # print(f"Eval images: {len(eval0.dataset)} | Batches (workers=0): {len(eval0)} | Batch size: {BATCH_SIZE}")
    # print(f"Eval images: {len(eval4.dataset)} | Batches (workers=4): {len(eval4)} | Batch size: {BATCH_SIZE}")

    extra_config = get_mixed_precision_config_w4a8_first_last_8bit()

    # -------- FP32 (workers=0) --------
    fp32_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    fp32_top1, fp32_top5 = evaluate_topk(fp32_model, eval0, device)
    fp32_runtime, fp32_thr, fp32_lat_ms = measure_inference_blockwise(fp32_model, eval0, device, warmup_steps=WARMUP_STEPS)
    fp32_mem = get_runtime_mem_mb_params_buffers(fp32_model)
    # fp32_ckpt = get_checkpoint_size_mb_state_dict(fp32_model)
    # fp32_param32 = get_theoretical_size_mb(fp32_model, 32)

    # -------- PTQ baseline (workers=0) --------
    q_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    q_base = prepare_by_platform(q_base, BackendType.Academic, extra_config).to(device)

    enable_calibration(q_base)
    with torch.no_grad():
        for x, _ in calib0:
            q_base(x.to(device, non_blocking=True))
    enable_quantization(q_base)

    base_top1, base_top5 = evaluate_topk(q_base, eval0, device)
    base_runtime, base_thr, base_lat_ms = measure_inference_blockwise(q_base, eval0, device, warmup_steps=WARMUP_STEPS)
    base_mem = get_runtime_mem_mb_params_buffers(q_base)
    base_ckpt = get_checkpoint_size_mb_state_dict(q_base)

    base_w4a8 = theoretical_w4a8_model_size_mb(q_base)

    # -------- PTQ + QuantDIR (workers=4) --------
    q_QuantDIR = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    q_QuantDIR = prepare_by_platform(q_QuantDIR, BackendType.Academic, extra_config).to(device)

    enable_calibration(q_QuantDIR)
    with torch.no_grad():
        for x, _ in calib4:
            q_QuantDIR(x.to(device, non_blocking=True))
    enable_quantization(q_QuantDIR)

    t0 = time.time()
    q_QuantDIR = apply_QuantDIR_to_layers_post_quant(
        fp32_model=fp32_model,
        q_model=q_QuantDIR,
        calib_loader=calib4,  # workers=4
        device=device,
        conv_layers=SELECTED_LAYERS["conv"],
        bn_layers=SELECTED_LAYERS["bn"],
        max_batches_conv=MAX_BATCHES_CONV,
        max_batches_bn=MAX_BATCHES_BN,
    )
    QuantDIR_one_time_s = time.time() - t0

    kal_top1, kal_top5 = evaluate_topk(q_QuantDIR, eval4, device)
    kal_runtime, kal_thr, kal_lat_ms = measure_inference_blockwise(q_QuantDIR, eval4, device, warmup_steps=WARMUP_STEPS)
    kal_mem = get_runtime_mem_mb_params_buffers(q_QuantDIR)
    # kal_ckpt = get_checkpoint_size_mb_state_dict(q_QuantDIR)

    kal_w4a8 = theoretical_w4a8_model_size_mb(q_QuantDIR)
    kal_qd_over = quantdir_overhead_mb(q_QuantDIR)
    kal_w4a8_plus_qd = theoretical_w4a8_plus_quantdir_mb(q_QuantDIR)

    os.makedirs("./deploy_ckpt", exist_ok=True)
    torch.save(q_QuantDIR.state_dict(), "./deploy_ckpt/w4a8_QuantDIR_deploy_state_dict_resnet18.pth")

    print("\n" + "=" * 130)
    print("SUMMARY (FP32+INT4 use workers=0; QuantDIR uses workers=4)")
    print("=" * 130)
    print(f"{'Method':<22} {'Top-1':>7} {'Top-5':>7} | {'Runtime(s)':>10} {'Throughput(img/s)':>18} {'Latency(ms/batch)':>18} | {'Mem(MB)':>8} {'Ckpt(MB)':>9}")
    print("-" * 130)
    print(f"{'FP32':<22} {fp32_top1:>7.2f} {fp32_top5:>7.2f} | {fp32_runtime:>10.2f} {fp32_thr:>18.1f} {fp32_lat_ms:>18.2f} ")
    print(f"{'W4A8 Baseline':<22} {base_top1:>7.2f} {base_top5:>7.2f} | {base_runtime:>10.2f} {base_thr:>18.1f} {base_lat_ms:>18.2f}")
    print(f"{'W4A8 + QuantDIR v2':<22} {kal_top1:>7.2f} {kal_top5:>7.2f} | {kal_runtime:>10.2f} {kal_thr:>18.1f} {kal_lat_ms:>18.2f}")
    print("-" * 130)
    print(f"One-time QuantDIR fit/insert cost (offline): {QuantDIR_one_time_s:.2f} s")
    print("=" * 130)

    print("\n" + "=" * 130)
    print("THEORETICAL SIZE REPORTING")
    print("=" * 130)
    # print(f"FP32(all@32b param-pack) = {fp32_param32:.2f} MB")
    print(f"W4A8 baseline (weights@4b + qparams) = {base_w4a8:.2f} MB")
    print(f"W4A8+QuantDIR: baseline={kal_w4a8:.2f} MB + QuantDIR_overhead={kal_qd_over:.4f} MB = {kal_w4a8_plus_qd:.2f} MB")
    print("=" * 130)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
