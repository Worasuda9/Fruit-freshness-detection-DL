import os
import random
import shutil
import torch


def split_dataset(source_dir, target_dir, fruits, classes, split_ratio):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    for split in split_ratio:
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

    for fruit in fruits:
        for cls in classes:
            folder = os.path.join(source_dir, fruit, cls)
            images = [img for img in os.listdir(folder) if not img.startswith('.')]
            random.shuffle(images)

            total = len(images)
            train_end = int(total * split_ratio["train"])
            val_end = train_end + int(total * split_ratio["val"])

            splits = {
                "train": images[:train_end],
                "val":   images[train_end:val_end],
                "test":  images[val_end:]
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total
