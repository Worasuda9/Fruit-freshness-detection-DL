# Fruit Freshness Classification

A PyTorch-based project that compares 3 models for classifying fruits as **fresh** or **rotten** across 3 fruit types: apple, banana, and orange.

---

## Dataset Download

| Data | Link |
|------|------|
| Full Dataset | [Download from Google Drive](https://drive.google.com/file/d/1SS-HpNd_HfUSEyjbPm77vwMb4BErP1z5/view?usp=drive_link) |
| Test Data | [Download from Google Drive](https://drive.google.com/drive/folders/1_ro7M60Mb1lRmGJEFkiZaQLWGbEVSlDr) |

---

## Dataset Structure

```
dataset/
├── apple/
│   ├── fresh/
│   └── rotten/
├── banana/
│   ├── fresh/
│   └── rotten/
└── orange/
    ├── fresh/
    └── rotten/
```

After splitting:

```
dataset_split/
├── train/   (70%)
├── val/     (15%)
└── test/    (15%)
```

---

## General Config (All Models)

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Device | CUDA / CPU (auto-detect) |
| Random Seed | 42 |

---

## Models

---

### Model 1 — Baseline CNN (No Augmentation)

**File:** `model1.py`

**Concept:** A minimal baseline using a simple CNN built from scratch with no data augmentation, to establish a performance floor for comparison.

**Architecture:**
```
Input (3, 224, 224)
→ Conv2d(3, 16) + ReLU + MaxPool2d      → (16, 112, 112)
→ Conv2d(16, 32) + ReLU + MaxPool2d     → (32, 56, 56)
→ Conv2d(32, 64) + ReLU + MaxPool2d     → (64, 28, 28)
→ Flatten → Linear(50176, 128) → ReLU
→ Linear(128, 2)
```

**Transform:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Pros:**
- Simple and easy to understand
- Good baseline for comparison

**Cons:**
- No augmentation → prone to overfitting on small datasets
- Learns from scratch → requires more data to generalize
- Small architecture → limited feature extraction capacity

---

### Model 2 — CNN + Data Augmentation

**File:** `model2.py`

**Concept:** Same SimpleCNN architecture as Model 1, but with data augmentation applied to the training set to reduce overfitting and improve generalization.

**Architecture:** Same as Model 1 (SimpleCNN)

**Train Transform:**
```python
transforms_v2.Compose([
    transforms_v2.Resize((256, 256)),
    transforms_v2.RandomCrop((224, 224)),
    transforms_v2.RandomHorizontalFlip(p=0.5),
    transforms_v2.RandomVerticalFlip(p=0.3),
    transforms_v2.RandomRotation(degrees=15),
    transforms_v2.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1),
    transforms_v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

**Val/Test Transform (no augmentation):**
```python
transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

**Pros:**
- Augmentation helps reduce overfitting
- Separate val/test transform ensures stable and realistic evaluation
- ColorJitter simulates color variation across different lighting conditions

**Cons:**
- Still uses SimpleCNN → limited feature extraction compared to deeper models
- Normalize uses `[0.5, 0.5, 0.5]` which differs from ImageNet stats

---

### Model 3 — ResNet18 (Transfer Learning)

**File:** `model3.py`

**Concept:** Uses a ResNet18 pretrained on ImageNet with all layers frozen except the final fully connected layer, which is replaced and fine-tuned for this binary classification task.

**Architecture:**
```
ResNet18 (pretrained on ImageNet, all layers frozen)
→ fc: Linear(512, 2)   ← only this layer is trained
```

**Transform:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # ImageNet stats
])
```

**Key Code:**
```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer only
model.fc = nn.Linear(model.fc.in_features, 2)

# Only optimize the fc layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

**Output:** Model weights saved as `fruit_model3_fast.pth`

**Pros:**
- Leverages pretrained ImageNet features → highest expected accuracy
- Trains faster since most layers are frozen
- Works well on small datasets

**Cons:**
- Fully frozen backbone → may underfit if dataset differs significantly from ImageNet
- No data augmentation applied

---

### Model 3-1 — ResNet18 (Optimized Variant)

**File:** `model3-1.py`

**Concept:** An optimized variant of Model 3 using ResNet18 with mixed precision training, larger batch size, and different image size for improved performance.

**Architecture:** Same as Model 3 (ResNet18 with frozen backbone, trainable fc layer)

**Key Differences from Model 3:**

- Image Size: 160 × 160 (vs 224 × 224)
- Batch Size: 64 (vs 32)
- Mixed Precision Training: Enabled for faster training on CUDA
- Weight Decay: 1e-4 added to optimizer
- Saves model as `fruit_model3_2_fast.pth`

**Transform:** Same as Model 3

**Key Code:**
```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)

# Mixed precision
scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# In training loop:
with torch.amp.autocast(enabled=(DEVICE.type == "cuda")):
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Output:** Model weights saved as `fruit_model3_2_fast.pth`

**Pros:**
- Mixed precision speeds up training
- Larger batch size for better gradient estimation
- Weight decay helps prevent overfitting

**Cons:**
- Smaller image size may lose some details
- More complex training loop

---

## Model Comparison

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Architecture | SimpleCNN | SimpleCNN | ResNet18 |
| Pretrained | ❌ | ❌ | ✅ ImageNet |
| Augmentation | ❌ | ✅ | ❌ |
| Normalization | ✅ ImageNet | ✅ [0.5] | ✅ ImageNet |
| Separate Val/Test Transform | ❌ | ✅ | ❌ |
| Save Model Weights | ❌ | ❌ | ✅ .pth |
| Expected Accuracy | Lowest | Medium | Highest |

---


## Output

All models produce the following outputs:
- Per-epoch Train/Val Loss and Accuracy printed to console
- Final Test Accuracy
- Confusion Matrix (evaluated on val set)
- Loss Curve and Accuracy Curve plots
- *(Model 3 and Model 3-1)* Saved model weights → `fruit_model3_fast.pth` and `fruit_model3_2_fast.pth` respectively

---

## Requirements

### Python Dependencies

Install required packages:
```
pip install -r requirements.txt
```

## Docker Setup

### Build Docker Image
```
docker build -t fruit-app .
```
### Run the Application
```
docker run -p 8501:8501 fruit-app
```
Then open your browser:
```
http://localhost:8501
```
