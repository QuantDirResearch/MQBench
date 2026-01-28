# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models                           # for example model
# from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
# from mqbench.prepare_by_platform import BackendType           # contain various Backend, contains Academic.
# from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
# from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
# from torch.utils.data import DataLoader 
# model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
# model.eval()
# backend = BackendType.Tensorrt
# model = prepare_by_platform(model, BackendType.Academic)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Load validation dataset
# val_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# val_dataset = torchvision.datasets.CIFAR100(root='shova_models/data', train=False, download=True, transform=val_transform)
# val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

# # Evaluate original model
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, targets in val_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

# fp32_accuracy = 100.0 * correct / total
# print(f"Original FP32 model accuracy: {fp32_accuracy:.2f}%")



import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from mqbench.prepare_by_platform import prepare_by_platform
from mqbench.prepare_by_platform import BackendType

# Step 1: Load and prepare model
print("Step 1: Loading and preparing model...")
# Use the new recommended way to load pretrained models
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Step 2: Prepare for quantization
print("Step 2: Preparing for quantization...")
model = prepare_by_platform(model, BackendType.Academic)

# Step 3: Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Step 3: Moving to device {device}")
model = model.to(device)

# Step 4: Prepare dataset
print("Step 4: Preparing dataset...")
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 5: Load dataset
print("Step 5: Loading dataset...")
try:
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform)
    print(f"Dataset loaded with {len(val_dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 6: Create data loader
print("Step 6: Creating data loader...")
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# Step 7: Modify model for CIFAR-100 (100 classes instead of 1000)
print("Step 7: Adapting model for CIFAR-100...")
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 100)  # CIFAR-100 has 100 classes
model = model.to(device)

# Step 8: Evaluate
print("Step 8: Starting evaluation...")
correct = 0
total = 0

try:
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{len(val_loader)}")
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print progress for first batch to verify things are working
            if batch_idx == 0:
                print(f"  First batch: {correct}/{targets.size(0)} correct")

    # Step 9: Print results
    print("Step 9: Calculating final results...")
    accuracy = 100.0 * correct / total
    print(f"Final accuracy: {accuracy:.2f}% ({correct}/{total})")
    
except Exception as e:
    print(f"Error during evaluation: {e}")