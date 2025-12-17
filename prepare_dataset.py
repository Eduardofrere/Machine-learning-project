from pathlib import Path
import pandas as pd
import random

# -------------
# CONFIG
# -------------

# Project root = folder where this script lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the CSV from the NIH dataset
# Adjust this if Data_Entry_2017.csv is inside a subfolder.
METADATA_CSV = PROJECT_ROOT / "data" / "Data_Entry_2017.csv"

# Output CSV with filtered 4-class labels
OUTPUT_CSV = PROJECT_ROOT / "data" / "four_class_labels.csv"

# Optional: how many images per class (set to None to keep all)
IMAGES_PER_CLASS = 200  # or None


# Classes we care about
TARGET_CLASSES = ["Normal", "Pneumonia", "Cardiomegaly", "Effusion"]

# Priority if there are multiple labels
PRIORITY_ORDER = ["Pneumonia", "Cardiomegaly", "Effusion", "Normal"]

# Map class name to numeric ID
CLASS_TO_ID = {
    "Normal": 0,
    "Pneumonia": 1,
    "Cardiomegaly": 2,
    "Effusion": 3,
}


def map_labels_to_four_classes(finding_labels: str):
    """
    Convert NIH 'Finding Labels' string (e.g. 'Cardiomegaly|Effusion')
    to one of: 'Normal', 'Pneumonia', 'Cardiomegaly', 'Effusion'
    according to a priority rule.

    Returns:
        - class name (str) if we keep this image
        - None if we want to discard the image
    """
    labels = [lbl.strip() for lbl in finding_labels.split("|")]

    # 'No Finding' means Normal
    if "No Finding" in labels:
        labels.append("Normal")

    # If none of our target labels are present, drop this sample
    if not any(lbl in labels for lbl in TARGET_CLASSES):
        return None

    # Apply priority
    for disease in PRIORITY_ORDER:
        if disease in labels:
            return disease

    return None  # safety fallback


def main():
    random.seed(42)

    print(f"Reading metadata from: {METADATA_CSV}")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {METADATA_CSV}. "
            f"Check that Data_Entry_2017.csv is in the 'data' folder."
        )

    df = pd.read_csv(METADATA_CSV)

    # Apply mapping to get a single 4-class label per image
    df["four_class_label"] = df["Finding Labels"].apply(map_labels_to_four_classes)

    # Drop rows we don't want (other diseases)
    df = df.dropna(subset=["four_class_label"]).reset_index(drop=True)

    print("Counts per class BEFORE balancing:")
    print(df["four_class_label"].value_counts())

    # Add numeric label id
    df["label_id"] = df["four_class_label"].map(CLASS_TO_ID)

    # Keep the image filename (we'll use this later in the Dataset)
    df["image_filename"] = df["Image Index"]

    # Optional: make a balanced subset
    if IMAGES_PER_CLASS is not None:
        balanced_parts = []
        for cls in TARGET_CLASSES:
            cls_df = df[df["four_class_label"] == cls]
            n_available = len(cls_df)
            if n_available == 0:
                print(f"Warning: no samples found for class {cls}")
                continue

            n_to_sample = min(IMAGES_PER_CLASS, n_available)
            print(f"Sampling {n_to_sample} rows for class {cls} (available: {n_available})")
            balanced_parts.append(cls_df.sample(n=n_to_sample, random_state=42))

        df_balanced = pd.concat(balanced_parts).reset_index(drop=True)
    else:
        df_balanced = df

    print("Final counts per class:")
    print(df_balanced["four_class_label"].value_counts())

    # Save filtered CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved filtered labels to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
