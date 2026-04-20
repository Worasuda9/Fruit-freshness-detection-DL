import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import split_dataset, calculate_accuracy

SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_split"

SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["fresh", "rotten"]
FRUITS = ["apple", "banana", "orange"]

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
torch.manual_seed(42)

# SPLIT DATA
split_dataset(SOURCE_DIR, TARGET_DIR, FRUITS, CLASSES, SPLIT_RATIO)
print("Dataset split completed!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# TRAIN LOOP
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            val_loss += criterion(model(images), labels).item()

    val_loss /= len(val_loader)

    train_acc = calculate_accuracy(model, train_loader, DEVICE)
    val_acc = calculate_accuracy(model, val_loader, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
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
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fresh", "Rotten"],
            yticklabels=["Fresh", "Rotten"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Model 3(ResNet18): Confusion Matrix")
plt.show()

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Model 3(ResNet18): Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Model 3(ResNet18): Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

torch.save(model.state_dict(), "fruit_model3_fast.pth")
print("Model saved as fruit_model3_fast.pth")
