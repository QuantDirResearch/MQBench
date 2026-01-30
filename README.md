# QuantDIR Reproducibility: RQ3 and RQ4

This README describes how to reproduce:
- **RQ3**: QuantDIR vs PTQ baselines — **accuracy and inference time**
- **RQ4**: Ablation study — layer-selection strategies (e.g., early / late / random)

---

## 1) Environment Setup (Python venv)

### 1.1 Python version

Use **Python 3.10+** (recommended). Check:
```bash
python3 --version
```

### 1.2 Create and activate a virtual environment

From the project root:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:
```bash
python -m pip install --upgrade pip
```

---

## 2) Install Dependencies

### 2.1 Install MQBench

Clone MQBench:
```bash
git clone https://github.com/ModelTC/MQBench.git
```

Install it editable:
```bash
cd MQBench
pip install -e .
cd ..
```

### 2.2 Install project requirements
```bash
pip install -r requirements.txt
```

**Note**: Ensure your installed torch and torchvision versions are compatible with MQBench and your scripts.

---

## 3) Dataset Preparation (ImageNet Validation)

### 3.1 Download ImageNet

Download ImageNet from the official site: https://www.image-net.org/

You need the ImageNet validation set (commonly `ILSVRC2012_img_val.tar`).

### 3.2 Extract validation images
```bash
mkdir -p data/test_dataset
tar -xf ILSVRC2012_img_val.tar -C data/test_dataset
```

### 3.3 Convert validation set to class-labeled subfolders

Use the provided `valprep.sh`:
```bash
bash valPrep.sh data/test_dataset
```

Expected structure:
```
data/imagenet/
  n01440764/
    ILSVRC2012_val_00000001.JPEG
    ...
  n01443537/
    ...
```

---

## 4) Reproduce RQ3

Make sure the image folder path is the same folder as the model and it is under `test_dataset` folder. Otherwise, add ImageNet path to `IMAGENET_ROOT` section of the code.
```bash
python3 QuantDIR_Resnet18.py  # for ResNet18
```

Same goes for other models too.

---

## 5) Reproduce RQ4
``` bash
Run the following command
python3 ablation_study_with_timing.py --resnet18 --early --n-select 24
```

## 6) Generate Instability Metric CSV Files

To generate the instability metrics, run the following command
 ``` bash
python3 instability_metrics.py --resnet18
```
Same goes for other models too
