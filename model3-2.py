import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils_v2 import split_dataset, calculate_accuracy

SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_split"

SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["fresh", "rotten"]
FRUITS = ["apple", "banana", "orange"]

BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 5

random.seed(42)
torch.manual_seed(42)

split_dataset(SOURCE_DIR, TARGET_DIR, FRUITS, CLASSES, SPLIT_RATIO)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(f"{TARGET_DIR}/train", transform=transform)
val_data = datasets.ImageFolder(f"{TARGET_DIR}/val", transform=transform)
test_data = datasets.ImageFolder(f"{TARGET_DIR}/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

def run_epoch(model, loader, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    running_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if optimizer is not None:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total

# =========================
# PHASE A: Freeze backbone
# =========================
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(EPOCH):
    train_loss, train_acc = run_epoch(model, train_loader, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\n[Phase A] Epoch {epoch+1}/{EPOCH}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# =========================
# PHASE B: Fine-tune all
# =========================
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

for epoch in range(EPOCH):
    train_loss, train_acc = run_epoch(model, train_loader, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\n[Phase B] Epoch {epoch+1}/{EPOCH}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# TEST
test_acc = calculate_accuracy(model, test_loader, DEVICE)
print(f"\nTest Accuracy: {test_acc:.4f}")

# CONFUSION MATRIX
all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
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
plt.title("Model 3 (ResNet18): Confusion Matrix")
plt.show()

epochs = list(range(1, len(train_losses) + 1))

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend()
plt.title("Model 3 (ResNet18): Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.legend()
plt.title("Model 3 (ResNet18): Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

torch.save(model.state_dict(), "fruit_model3_finetuned.pth")
print("Model saved as fruit_model3_finetuned.pth")