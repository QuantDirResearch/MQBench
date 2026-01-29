import os
import gc
import re
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# MQBench
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization


# ============================================================
# PATHS AND DATA SETUP
# ============================================================
imagenet_root = "./ILSVRC2012_img_val"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=imagenet_root, transform=transform)
calib_subset = Subset(dataset, range(0, 512))
test_subset  = Subset(dataset, range(0, 5000))

calib_loader = DataLoader(calib_subset, batch_size=64, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_subset,  batch_size=64, shuffle=False, num_workers=0)


# ============================================================
# HELPERS
# ============================================================
def flatten_for_svd(w: torch.Tensor) -> np.ndarray:
    w = w.detach().cpu().numpy()
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)
    return w

def cond_number(w: torch.Tensor) -> float:
    A = flatten_for_svd(w)
    if A.size == 0:
        return np.nan
    return float(np.linalg.cond(A))

def variance(w: torch.Tensor) -> float:
    return float(torch.var(w.detach().cpu()))

def cosine_similarity(w_fp: torch.Tensor, w_q: torch.Tensor) -> float:
    # Robust: avoid deprecated scalar conversions
    a = w_fp.detach().cpu().numpy().ravel().astype(np.float32, copy=False)
    b = w_q.detach().cpu().numpy().ravel().astype(np.float32, copy=False)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    #return 1.0 - sim
    return sim

def _manual_asymmetric_quantize_per_tensor_minmax(w: torch.Tensor, bit=4) -> torch.Tensor:
    w = w.detach().cpu()
    qmin, qmax = 0, (1 << bit) - 1
    w_min, w_max = torch.min(w), torch.max(w)
    if float(w_max - w_min) < 1e-12:
        return w.clone()
    scale = (w_max - w_min) / float(qmax - qmin)
    zp = qmin - torch.round(w_min / scale)
    q = torch.clamp(torch.round(w / scale + zp), qmin, qmax)
    dq = (q - zp) * scale
    return dq

def get_quant_weight(m: nn.Module, bit=4) -> torch.Tensor:
    if hasattr(m, "weight_fake_quant"):
        try:
            return m.weight_fake_quant(m.weight)
        except Exception:
            pass
    return _manual_asymmetric_quantize_per_tensor_minmax(m.weight, bit=bit)

# --- activation metrics ---
def rel_error(fp: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.norm(fp - q) / (torch.norm(fp) + 1e-12))

def snr(fp: torch.Tensor, q: torch.Tensor) -> float:
    s_pow = torch.var(fp).item()
    n_pow = torch.mean((fp - q) ** 2).item()
    return 10.0 * np.log10(max(s_pow, 1e-12) / max(n_pow, 1e-12))



def kl_div(fp: torch.Tensor, q: torch.Tensor, bins: int = 256, eps: float = 1e-8) -> float:
    
    # Flatten and bring to CPU float32 for stable histogramming
    fp_np = fp.detach().to(dtype=torch.float32, device="cpu").view(-1).numpy()
    q_np  = q.detach().to(dtype=torch.float32, device="cpu").view(-1).numpy()

    # Determine a shared support to avoid disjoint bins
    lo = float(min(fp_np.min(), q_np.min()))
    hi = float(max(fp_np.max(), q_np.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    # Histograms over the same edges
    hist_fp, edges = np.histogram(fp_np, bins=bins, range=(lo, hi))
    hist_q, _      = np.histogram(q_np,  bins=edges)

    # Convert to probabilities with epsilon for empty bins
    P = hist_fp.astype(np.float32) + eps
    Q = hist_q.astype(np.float32)  + eps
    P /= P.sum()
    Q /= Q.sum()

    kl = float(np.sum(P * (np.log(P) - np.log(Q))))
    return kl if kl >= 0.0 else 0.0


def mse_error(fp: torch.Tensor, q: torch.Tensor) -> float:
    fp_flat = fp.detach().cpu().float().flatten()
    q_flat = q.detach().cpu().float().flatten()
    n = min(fp_flat.numel(), q_flat.numel())
    if n == 0:
        return float('nan')
    diff = fp_flat[:n] - q_flat[:n]
    return float((diff * diff).mean().item())

def sign_flip_rate(fp: torch.Tensor, q: torch.Tensor) -> float:
    return float((fp.sign() != q.sign()).float().mean())



def magnitude_loss(fp: torch.Tensor, q: torch.Tensor, tau: float | None = None) -> float:
    fp_abs = fp.detach().abs()
    q_abs  = q.detach().abs()

    if tau is None:
        med = torch.median(fp_abs)
        tau = 1e-6 if not torch.isfinite(med) or med == 0 else float(med) * 1e-3

    fp_sig = fp_abs > tau      # meaningful in FP32
    q_zero = q_abs <= tau      # killed by quantization

    count = (fp_sig & q_zero).sum().item()
    total_meaningful = fp_sig.sum().item()

    if total_meaningful == 0:
        return 0.0

    # Return RATIO, not raw count
    return count / total_meaningful   # value between 0.0 and 1.0


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# ============================================================
# DETECTION (no file I/O; returns DataFrame)
# ============================================================
def layer_analysis_detection(quant_model: nn.Module, calib_loader, num_batches=10, device=None) -> pd.DataFrame:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Layer Analysis: Detection (Weights + BN/ReLU) ===")
    quant_model.eval().to(device)

    # Step 1: execution order
    execution_order, handles = [], []
    def track_execution_order(name):
        def hook(m, i, o):
            if name not in execution_order:
                execution_order.append(name)
        return hook

    interested = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm,
                  nn.ReLU, nn.ReLU6, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)

    for name, m in quant_model.named_modules():
        if isinstance(m, interested):
            handles.append(m.register_forward_hook(track_execution_order(name)))

    with torch.no_grad():
        dummy_batch = next(iter(calib_loader))[0][:1].to(device)
        quant_model(dummy_batch)
    for h in handles:
        h.remove()

    print(f"Found {len(execution_order)} layers in execution order")

    # Step 2: ReLU dead-neuron ratio
    relu_outputs = {}
    def hook_relu(name):
        def fn(module, input, output):
            relu_outputs.setdefault(name, []).append(output.detach().cpu())
        return fn

    handles = []
    for name, m in quant_model.named_modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)):
            handles.append(m.register_forward_hook(hook_relu(name)))

    with torch.no_grad():
        for bidx, (images, _) in enumerate(calib_loader):
            images = images.to(device)
            quant_model(images)
            if bidx >= num_batches - 1:
                break
    for h in handles:
        h.remove()

    # Step 3: metrics
    results, count = [], 0
    modules_dict = dict(quant_model.named_modules())

    for name in execution_order:
        if name not in modules_dict:
            continue
        m = modules_dict[name]

        cond32 = cond4 = var32 = var4 = cosine_sim = None
        dead_neuron = None

        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            w_fp = m.weight
            w_q  = get_quant_weight(m, bit=4)
            var32 = variance(w_fp)
            var4  = variance(w_q)
            cond32 = cond_number(w_fp)
            cond4  = cond_number(w_q)
            cosine_sim = cosine_similarity(w_fp, w_q)

        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)) and hasattr(m, "weight"):
            w_fp = m.weight
            w_q  = get_quant_weight(m, bit=4)
            var32 = variance(w_fp)
            var4  = variance(w_q)
            cosine_sim = cosine_similarity(w_fp, w_q)
            if w_fp.ndim == 1:
                cond32 = float('nan'); cond4 = float('nan')
            else:
                cond32 = cond_number(w_fp); cond4 = cond_number(w_q)

        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            if name in relu_outputs:
                acts = torch.cat(relu_outputs[name], dim=0)
                dead_neuron = float((acts == 0).float().mean())

        count += 1
        results.append({
            "idx": count,
            "layer": name,
            "type": m.__class__.__name__,
            "cond_32": cond32,
            "cond_4":  cond4,
            "var_32":  var32,
            "var_4":   var4,
            "cosine_sim": cosine_sim,
            "dead_neuron_ratio": dead_neuron,
        })

    df = pd.DataFrame(results)
    print(f"Total layers analyzed: {count}")
    return df


# ============================================================
# PROPAGATION (no file I/O; returns DataFrame)
# ============================================================
def layer_analysis_propagation(fp32_model: nn.Module, quant_model: nn.Module, loader,
                               num_batches=3, device=None) -> pd.DataFrame:
    """
    Layer-wise propagation analysis between FP32 and quantized models.

    Metrics:
      - RelErr: total relative error vs FP32 at this layer
      - error_delta: cumulative error growth (RelErr_L - RelErr_{L-1})
      - error_propagation: true amplification factor:
            error_in  = RelErr between previous FP32 vs quant outputs
            error_out = RelErr at current layer
            error_propagation = error_out / (error_in + eps), NaN for first layer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fp32_model.eval().to(device)
    quant_model.eval().to(device)

    # execution order (from fp32)
    execution_order, handles = [], []
    def track_execution_order(name):
        def hook(m, i, o):
            if name not in execution_order:
                execution_order.append(name)
        return hook

    interested = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm,
                  nn.ReLU, nn.ReLU6, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
    for name, m in fp32_model.named_modules():
        if isinstance(m, interested):
            handles.append(m.register_forward_hook(track_execution_order(name)))
    with torch.no_grad():
        dummy_batch = next(iter(loader))[0][:1].to(device)
        fp32_model(dummy_batch)
    for h in handles:
        h.remove()
    print(f"Found {len(execution_order)} layers in execution order")

    results = []
    prev_relerr = None
    prev_fp32_output = None
    prev_quant_output = None
    eps = 1e-12

    modules_fp32 = dict(fp32_model.named_modules())
    modules_quant = dict(quant_model.named_modules())

    def flatten_or_gap(tensors):
        outs = []
        for x in tensors:
            if x.ndim == 4:
                x = x.mean(dim=[2, 3])
            elif x.ndim > 2:
                x = x.flatten(start_dim=1)
            outs.append(x)
        return torch.cat(outs, dim=0)

    for idx, name in enumerate(execution_order):
        if name not in modules_fp32 or name not in modules_quant:
            continue

        fp32_outs, quant_outs = [], []
        def hook_fp32(m, i, o): fp32_outs.append(o.detach())
        def hook_quant(m, i, o): quant_outs.append(o.detach())
        h1 = modules_fp32[name].register_forward_hook(hook_fp32)
        h2 = modules_quant[name].register_forward_hook(hook_quant)

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                images = images.to(device, non_blocking=True)
                _ = fp32_model(images)
                _ = quant_model(images)

        h1.remove(); h2.remove()
        if not fp32_outs or not quant_outs:
            continue

        # keep majority shape per side
        shapes_fp = [x.shape for x in fp32_outs]
        shapes_q  = [x.shape for x in quant_outs]
        common_shape_fp = max(set(shapes_fp), key=shapes_fp.count)
        common_shape_q  = max(set(shapes_q),  key=shapes_q.count)
        fp32_valid  = [x.cpu() for x in fp32_outs  if x.shape == common_shape_fp]
        quant_valid = [x.cpu() for x in quant_outs if x.shape == common_shape_q]
        if not fp32_valid or not quant_valid:
            continue

        fp32_acts  = flatten_or_gap(fp32_valid)
        quant_acts = flatten_or_gap(quant_valid)

        # align dims
        n = min(fp32_acts.shape[0], quant_acts.shape[0])
        m = min(fp32_acts.shape[1], quant_acts.shape[1])
        fp32_acts  = fp32_acts[:n, :m]
        quant_acts = quant_acts[:n, :m]

        layer_type = modules_fp32[name].__class__.__name__

        # metrics
        relerr    = rel_error(fp32_acts, quant_acts)
        snr_val   = snr(fp32_acts, quant_acts)
        kl_val    = kl_div(fp32_acts, quant_acts)
        mse_val   = mse_error(fp32_acts, quant_acts)
        mag_loss  = magnitude_loss(fp32_acts, quant_acts)
        sfr       = sign_flip_rate(fp32_acts, quant_acts)

        inherited_error = float('nan') if prev_relerr is None else prev_relerr
        cumulative_error_growth = relerr - (prev_relerr if prev_relerr is not None else 0.0)
    
        if prev_fp32_output is not None and prev_quant_output is not None:
            error_in  = rel_error(prev_fp32_output, prev_quant_output)
            error_out = relerr
            true_amplification = error_out / (error_in + eps) if error_in > eps else 1.0
        else:
            true_amplification = float('nan')

        # Store for next layer
        prev_fp32_output  = fp32_acts.clone().cpu()
        prev_quant_output = quant_acts.clone().cpu()
        prev_relerr = relerr

        results.append({
            "idx": idx + 1,
            "layer": name,
            "type": layer_type,
            "SNR": snr_val,
            "KL": kl_val,
            "SFR": sfr,
            "magnitude_loss": mag_loss,
            "RelErr": relerr,
            "mse_error": mse_val,
            "inherited_error": inherited_error,
            "error_delta": cumulative_error_growth,
            "error_propagation": true_amplification,     
        })

        del fp32_outs, quant_outs, fp32_acts, quant_acts
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    return df

# ============================================================
# FULL per-layer sensitivity (W4 weights + A8 outputs; A8-only for BN/ReLU/Pool)
# ============================================================
def _quant_dequant_tensor(x: torch.Tensor, bit=8, symmetric=False):
    x = x.detach()
    if symmetric:
        maxv = x.abs().max()
        scale = max(maxv / (2**(bit-1) - 1), 1e-12)
        q = torch.clamp(torch.round(x / scale), -(2**(bit-1)), 2**(bit-1) - 1)
        return (q * scale).to(x.dtype)
    else:
        xmin, xmax = x.min(), x.max()
        if float(xmax - xmin) < 1e-12:
            return x
        qmin, qmax = 0, 2**bit - 1
        scale = (xmax - xmin) / float(qmax - qmin)
        zp = qmin - torch.round(xmin / scale)
        q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
        return ((q - zp) * scale).to(x.dtype)

def _quantize_layer_weights_inplace(module: nn.Module, wbit=4, symmetric=False):
    if not hasattr(module, "weight") or module.weight is None:
        return
    w = module.weight.data
    if symmetric:
        maxv = w.abs().max()
        scale = max(maxv / (2**(wbit-1) - 1), 1e-12)
        q = torch.clamp(torch.round(w / scale), -(2**(wbit-1)), 2**(wbit-1) - 1)
        module.weight.data = (q * scale).to(w.dtype)
    else:
        wmin, wmax = w.min(), w.max()
        if float(wmax - wmin) < 1e-12:
            return
        qmin, qmax = 0, 2**wbit - 1
        scale = (wmax - wmin) / float(qmax - qmin)
        zp = qmin - torch.round(wmin / scale)
        q = torch.clamp(torch.round(w / scale + zp), qmin, qmax)
        module.weight.data = ((q - zp) * scale).to(w.dtype)

def make_model_quant_fn_W4A8(include_nonparametric=True, wbit=4, abit=8):
    def model_quant_fn(model, target_name):
        modules = dict(model.named_modules())
        m = modules.get(target_name, None)
        if m is None:
            return model

        # Conv / Linear: W4 + A8
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _quantize_layer_weights_inplace(m, wbit=wbit, symmetric=False)
            orig_forward = m.forward
            def wrapped_forward(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                return _quant_dequant_tensor(out, bit=abit, symmetric=False)
            m.forward = wrapped_forward
            return model

        # BN / ReLU / Pool: A8 (activation quant only)
        if include_nonparametric and isinstance(
            m, (nn.BatchNorm2d, nn.LayerNorm, nn.ReLU, nn.ReLU6,
                nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
        ):
            orig_forward = m.forward
            def wrapped_forward(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                return _quant_dequant_tensor(out, bit=abit, symmetric=False)
            m.forward = wrapped_forward
            return model

        return model
    return model_quant_fn

def layerwise_quantization_sensitivity_all(model_fp32: nn.Module, dataloader, eval_fn,
                                           wbit=4, abit=8, include_nonparametric=True):
    model_fp32.eval()
    base_acc = float(eval_fn(model_fp32, dataloader))
    model_quant_fn = make_model_quant_fn_W4A8(include_nonparametric=include_nonparametric,
                                              wbit=wbit, abit=abit)

    rows = []
    for name, module in model_fp32.named_modules():
        if name == "" or isinstance(module, (nn.Sequential,)):
            continue
        mcopy = copy.deepcopy(model_fp32).eval()
        mcopy = model_quant_fn(mcopy, name)
        acc_q = float(eval_fn(mcopy, dataloader))
        rows.append({"layer": name, "sensitivity_W4A8": base_acc - acc_q})

    df_sens = pd.DataFrame(rows).sort_values("sensitivity_W4A8", ascending=False).reset_index(drop=True)
    return df_sens

def prepare_rf_data(df_all: pd.DataFrame):
    
    # 1. Define Features (X) and Target (y)
    # Target: The ground truth sensitivity from the slow loop
    target_col = "sensitivity_W4A8" 
    
    # Features: Only use metrics available from the FAST analysis (Detection + Propagation)
    feature_cols = [
        "cond_4", "cosine_sim", "dead_neuron_ratio", "SNR", "KL", "SFR", 
        "magnitude_loss", "RelErr", "mse_error", "inherited_error", "error_delta", 
        "error_propagation"
    ]
    
    # Remove layers that couldn't be evaluated or have missing target data
    df_clean = df_all.dropna(subset=[target_col])
    
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].values
    
    # 2. Impute and Log Transform
    # Impute missing values with median for robustness
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0) 
    
    # Apply Log Transform to stabilize large-scale metrics
    for col in ["cond_4", "mse_error", "inherited_error", "error_delta"]:
        if col in X.columns:
            # Use log10(abs(x) + 1e-12) for stability and to handle zeros
            X[col] = np.log10(np.abs(X[col]) + 1e-12) 
            # Re-impute potential log(-inf) if all values were zero
            X[col] = X[col].replace([np.inf, -np.inf], 0.0)

    # 3. Scale Features (Essential for comparing metrics)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    return pd.DataFrame(X_scaled, columns=X.columns), y, X.columns.tolist()


def run_shap_analysis(df_metrics: pd.DataFrame):
    print("\n[7] Running SHAP analysis (Predicting sensitivity_W4A8)...")
    
    # --- STEP 1: PREPARE DATA ---
    X_scaled_df, y, feature_cols = prepare_rf_data(df_metrics)
    
    if len(feature_cols) == 0:
        out = df_metrics[["layer"]].copy()
        out["instability_shap"] = np.nan
        return out, {}
    
    X_scaled = X_scaled_df.values
    
    print(f"[INFO] RF Target: sensitivity_W4A8 (Ground Truth Drop)")
    print(f"[INFO] Features used: {len(feature_cols)}")
    
    # --- STEP 2: TRAIN RF MODEL ---
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    # X_scaled has already been prepared and scaled
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)

    # --- STEP 3: SHAP VALUE CALCULATION ---
    shap_importance = {}
    sv = None
    try:
        import shap
        # Using the scaled features for explainer input
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled) 
        
        # Calculate mean absolute SHAP value for feature importance
        shap_importance = dict(zip(feature_cols, np.abs(sv).mean(axis=0)))
        
        print("\nTop SHAP features (mean |SHAP|):")
        for k, v in sorted(shap_importance.items(), key=lambda x: -x[1])[:10]:
            # The name 'cond_4' now represents the log-transformed value
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"[INFO] SHAP calculation failed (Is the 'shap' library installed?): {e}")

    # --- STEP 4: OUTPUT RESULTS ---
    out = df_metrics[["layer"]].copy()
    
    # 'instability_shap' is now the RF's prediction of the true sensitivity
    out["instability_shap"] = preds
    
    if sv is not None:
        # shap_magnitude: Total impact magnitude for that layer (Sum of all feature impacts)
        out["shap_magnitude"] = np.abs(sv).sum(axis=1)
    
    return out, shap_importance
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FP32 baseline
    print("\n[1] Loading FP32 model...")
    # fp32_model = models.mobilenet_V3_small(pretrained=True).eval()
    # fp32_model = models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
    # fp32_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
    # fp32_model = models.resnet34(weights="ResNet34_Weights.DEFAULT").eval()
    fp32_model = models.regnet_x_400mf(weights="DEFAULT").eval()
    acc_fp32 = evaluate(fp32_model, test_loader)
    print(f"FP32 model accuracy: {acc_fp32:.2f}%")

    # Quantized model (MQBench Academic backend)
    print("\n[2] Preparing quantized model...")
    # quant_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
    # quant_model = models.resnet34(weights="ResNet34_Weights.DEFAULT").eval()
    # quant_model = models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
    quant_model = models.regnet_x_400mf(weights="DEFAULT").eval()
    extra_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MinMaxObserver',#'MSEObserver',
            'a_observer': 'EMAMinMaxObserver', #'EMAMSEObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
            'w_qscheme': {
                'bit': 4,
                'symmetry': False,
                'per_channel': True,
                'pot_scale': False,
            },
            'a_qscheme': {
                'bit': 8,
                'symmetry': False,
                'per_channel': False,
                'pot_scale': False,
            }
        }
    }
    quant_model = prepare_by_platform(quant_model, BackendType.Academic, extra_config)

    print("[3] Calibrating...")
    enable_calibration(quant_model)
    with torch.no_grad():
        for images, _ in calib_loader:
            quant_model(images)
    enable_quantization(quant_model)

    acc_quant = evaluate(quant_model, test_loader)
    print(f"Quantized model accuracy: {acc_quant:.2f}%")
    print(f"Accuracy drop: {acc_fp32 - acc_quant:.2f}%")

    print("\n[4] Running layer analysis...")
    df_detection   = layer_analysis_detection(quant_model, calib_loader, device=device)
    df_propagation = layer_analysis_propagation(fp32_model, quant_model, test_loader,
                                                num_batches=5, device=device)

    print("\n[5] Merging detection & propagation into a single DataFrame...")
    df_det  = df_detection.copy()
    df_prop = df_propagation.copy()

    # Clean detection layer names (_dupN) and dedup
    df_det["layer"] = df_det["layer"].apply(lambda s: re.sub(r"_dup\d+$", "", s))
    df_det = df_det.sort_values("idx").drop_duplicates(subset=["layer"], keep="first")

    prop_keys = {"layer", "idx", "type"}
    prop_cols = ["layer"] + [c for c in df_prop.columns if c not in prop_keys]

    # Base df_all = detection ⨝ propagation
    df_all = df_det.merge(df_prop[prop_cols], on="layer", how="left")
    df_all = df_all.sort_values("idx").reset_index(drop=True)
    df_all["idx"] = np.arange(1, len(df_all) + 1)

    print("\n[6] Running FULL per-layer sensitivity (W4 weights + A8 output; A8-only BN/ReLU/Pool)...")
    df_sens = layerwise_quantization_sensitivity_all(
        model_fp32=fp32_model,
        dataloader=test_loader,
        eval_fn=evaluate,
        wbit=4,
        abit=8,
        include_nonparametric=True
    )

    # Now merge sensitivity_W4A8 into df_all
    df_all = df_all.merge(df_sens, on="layer", how="left")

    print("\n[7] Running SHAP analysis (Predicting sensitivity_W4A8)...")
    shap_df, shap_importance = run_shap_analysis(df_all)
    df_all = df_all.merge(shap_df, on="layer", how="left")

    
    # ---- Single CSV output
    os.makedirs("ptq_metrics", exist_ok=True)
    model_name = fp32_model.__class__.__name__.lower()
    combined_path = f"ptq_metrics/instability_metrics_{model_name}600.csv"
    df_all.to_csv(combined_path, index=False)

    print(f"[OK] Saved merged CSV -> {os.path.abspath(combined_path)}")
    print(f"[OK] Layers in detection (after dedup): {len(df_det)}")
    print(f"[OK] Layers in final merged CSV:        {len(df_all)}")

    # Optional: print full sensitivity list
    # sens_list = list(zip(df_sens["layer"].tolist(),
    #                      np.round(df_sens["sensitivity_W4A8"].values, 3).tolist()))
    # print("\nFull per-layer sensitivity (W4A8):")
    # print(sens_list)
