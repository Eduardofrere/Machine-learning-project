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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# -----------------------
# CONFIG
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "four_class_labels.csv"
IMAGES_ROOT = PROJECT_ROOT / "data"

NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Pick best available device
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
            raise FileNotFoundError(f"Image file {filename} not found.")

        img_path = self.image_index[filename]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -----------------------
# MODEL
# -----------------------

def create_model(num_classes: int):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# -----------------------
# TRAINING / EVAL
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

    return running_loss / total, running_corrects / total


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

    return running_loss / total, running_corrects / total


# -----------------------
# MAIN
# -----------------------

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Run prepare_dataset.py first.")

    df = pd.read_csv(CSV_PATH)

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label_id"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

    print("Dataset sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    # Image index
    image_index = build_image_index(IMAGES_ROOT)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Datasets and loaders
    train_dataset = ChestXrayDataset(train_df, image_index, transform=train_transform)
    val_dataset = ChestXrayDataset(val_df, image_index, transform=eval_transform)
    test_dataset = ChestXrayDataset(test_df, image_index, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Loss
    class_counts = train_df["label_id"].value_counts().sort_index()
    class_weights = torch.tensor((1.0 / class_counts).values, dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = create_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Metric storage
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Training loop
    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
              f"- {time.time() - start:.1f}s - "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"Best val accuracy: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    # Save best model
    torch.save(model.state_dict(), PROJECT_ROOT / "best_resnet18_chestxray.pth")

    # -----------------------
    # PLOT TRAINING CURVES
    # -----------------------
    epochs = range(1, NUM_EPOCHS + 1)

    # Accuracy plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()

    # Loss plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    print("Saved accuracy_curve.png and loss_curve.png")

    # -----------------------
    # CONFUSION MATRIX
    # -----------------------
    print("Generating confusion matrix...")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    classes = ["Normal", "Pneumonia", "Cardiomegaly", "Effusion"]

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    main()
