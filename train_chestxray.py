import os
from pathlib import Path
import time
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# -----------------------
# CONFIG
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "four_class_labels.csv"
IMAGES_ROOT = PROJECT_ROOT / "data"   # we will search recursively here

NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Pick best available device (Apple GPU > CUDA > CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# -----------------------
# UTIL: BUILD IMAGE INDEX
# -----------------------

def build_image_index(images_root: Path):
    """
    Walk through images_root and build a mapping:
        filename -> full absolute path
    Works even if images are spread across subfolders.
    """
    index = {}
    for root, _, files in os.walk(images_root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                index[f] = os.path.join(root, f)
    if not index:
        raise RuntimeError(f"No images found under {images_root}")
    print(f"Indexed {len(index)} image files.")
    return index


# -----------------------
# DATASET
# -----------------------

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_index: dict, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_filename"]
        label = int(row["label_id"])

        if filename not in self.image_index:
            raise FileNotFoundError(f"Image file {filename} not found in image index.")

        img_path = self.image_index[filename]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -----------------------
# MODEL: RESNET18 TRANSFER LEARNING
# -----------------------

def create_model(num_classes: int):
    # Use ResNet-18 pretrained on ImageNet
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# -----------------------
# TRAINING / EVAL LOOPS
# -----------------------

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


# -----------------------
# MAIN
# -----------------------

def main():
    # 1. Load CSV
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}. "
                                f"Run prepare_dataset.py first.")
    df = pd.read_csv(CSV_PATH)

    # 2. Train / val / test split (stratified)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label_id"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_id"],
        random_state=42,
    )

    print("Dataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # 3. Build image index
    image_index = build_image_index(IMAGES_ROOT)

    # 4. Data transforms (augmentation for train)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 5. Datasets and loaders
    train_dataset = ChestXrayDataset(train_df, image_index, transform=train_transform)
    val_dataset = ChestXrayDataset(val_df, image_index, transform=eval_transform)
    test_dataset = ChestXrayDataset(test_df, image_index, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 6. Class-balanced loss (handles class imbalance)
    class_counts = train_df["label_id"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32).to(DEVICE)

    print("Class counts:", class_counts.to_dict())
    print("Class weights:", class_weights.to_dict())

    model = create_model(NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 7. Training loop
    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
              f"- {elapsed:.1f}s - "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"| Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Keep best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"Best val accuracy: {best_val_acc:.4f}")

    # 8. Evaluate best model on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    # 9. Save best model
    out_path = PROJECT_ROOT / "best_resnet18_chestxray.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved best model weights to: {out_path}")


if __name__ == "__main__":
    main()
