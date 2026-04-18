import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# =========================
# 1. CONFIG
# =========================
SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_split"

SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["fresh", "rotten"]
FRUITS = ["apple", "banana", "orange"]

BATCH_SIZE = 64
IMG_SIZE = 160
EPOCHS = 10
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. SEED
# =========================
random.seed(42)
torch.manual_seed(42)

# =========================
# 3. CLEAN SPLIT
# =========================
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

# =========================
# 4. CREATE SPLIT
# =========================
valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for split in SPLIT_RATIO:
    for cls in CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

for fruit in FRUITS:
    for cls in CLASSES:
        folder = os.path.join(SOURCE_DIR, fruit, cls)
        images = [
            img for img in os.listdir(folder)
            if not img.startswith('.') and img.lower().endswith(valid_ext)
        ]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * 0.7)
        val_end = train_end + int(total * 0.15)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split in splits:
            for img in splits[split]:
                src = os.path.join(folder, img)
                dst = os.path.join(TARGET_DIR, split, cls, f"{fruit}_{img}")
                shutil.copy(src, dst)

print("✅ Dataset split completed!")

# =========================
# 5. TRANSFORMS
# =========================
weights = ResNet18_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 6. LOAD DATA
# =========================
train_data = datasets.ImageFolder(f"{TARGET_DIR}/train", transform=transform)
val_data = datasets.ImageFolder(f"{TARGET_DIR}/val", transform=transform)
test_data = datasets.ImageFolder(f"{TARGET_DIR}/test", transform=transform)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    # persistent_workers=(NUM_WORKERS > 0)
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    # persistent_workers=(NUM_WORKERS > 0)
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    # persistent_workers=(NUM_WORKERS > 0)
)

# =========================
# 7. LOAD PRETRAINED MODEL
# =========================
model = resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# =========================
# 8. LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# mixed precision
amp_device = "cuda" if DEVICE.type == "cuda" else "cpu"
scaler = torch.amp.GradScaler(amp_device, enabled=(DEVICE.type == "cuda"))

# =========================
# 9. EVALUATION FUNCTION
# =========================
def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# =========================
# 10. TRAIN LOOP
# =========================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(amp_device, enabled=(DEVICE.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    val_loss, val_acc = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# =========================
# 11. TEST
# =========================
test_loss, test_acc = evaluate(test_loader)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# =========================
# 12. CONFUSION MATRIX
# =========================
all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fresh", "Rotten"],
            yticklabels=["Fresh", "Rotten"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 13. SAVE MODEL
# =========================
torch.save(model.state_dict(), "fruit_model3_fast.pth")
print("✅ Model saved as fruit_model3_fast.pth")
