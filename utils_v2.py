import os
import random
import shutil
import torch


def split_dataset(source_dir, target_dir, fruits, classes, split_ratio, force_resplit=False):
    if os.path.exists(target_dir):
        if force_resplit:
            shutil.rmtree(target_dir)
        else:
            print(f"Using existing split dataset: {target_dir}")
            return

    for split in split_ratio:
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for fruit in fruits:
        for cls in classes:
            folder = os.path.join(source_dir, fruit, cls)
            images = [
                img for img in os.listdir(folder)
                if not img.startswith('.') and img.lower().endswith(valid_ext)
            ]
            random.shuffle(images)

            total = len(images)
            train_end = int(total * split_ratio["train"])
            val_end = train_end + int(total * split_ratio["val"])

            splits = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:]
            }

            for split_name in splits:
                for img in splits[split_name]:
                    src = os.path.join(folder, img)
                    dst = os.path.join(target_dir, split_name, cls, f"{fruit}_{img}")
                    shutil.copy(src, dst)


def calculate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total