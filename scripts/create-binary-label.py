import argparse
import os

import pandas as pd
from tqdm import tqdm

ROOT_DIR = "../DAiSEE"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Processed")


def convert_to_binary_labels(df):
    """
    Convert multi-class labels to binary labels using unified mapping:
    {0,1} → 0 (low), {2,3} → 1 (high)
    - Boredom: 0->0, 1->0, 2->1, 3->1
    - Engagement: 0->0, 1->0, 2->1, 3->1
    - Confusion: 0->0, 1->0, 2->1, 3->1
    - Frustration: 0->0, 1->0, 2->1, 3->1
    """
    df_binary = df.copy()

    # Boredom: 0->0, 1->0, 2->1, 3->1
    df_binary["Boredom"] = df_binary["Boredom"].apply(lambda x: 1 if x >= 2 else 0)

    # Engagement: 0->0, 1->0, 2->1, 3->1
    df_binary["Engagement"] = df_binary["Engagement"].apply(
        lambda x: 1 if x >= 2 else 0
    )

    # Confusion: 0->0, 1->0, 2->1, 3->1
    df_binary["Confusion"] = df_binary["Confusion"].apply(lambda x: 1 if x >= 2 else 0)

    # Frustration: 0->0, 1->0, 2->1, 3->1
    df_binary["Frustration"] = df_binary["Frustration"].apply(
        lambda x: 1 if x >= 2 else 0
    )

    return df_binary


def process_object_labels(obj):
    """Process labels for a specific object (Train, Test, Validation)"""
    # Read the original labels
    labels_path = os.path.join(PROCESSED_DIR, f"{obj}_labels.csv")
    if not os.path.exists(labels_path):
        print(f"Warning: {labels_path} not found. Skipping {obj}.")
        return

    print(f"Processing {obj} labels from {labels_path}")
    df = pd.read_csv(labels_path)

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    print(f"Original {obj} dataset: {len(df)} samples")
    print("Original label distribution:")
    for col in ["Boredom", "Engagement", "Confusion", "Frustration"]:
        if col in df.columns:
            print(f"  {col}: {df[col].value_counts().sort_index().to_dict()}")

    # Convert to binary labels
    df_binary = convert_to_binary_labels(df)

    print("Binary label distribution:")
    for col in ["Boredom", "Engagement", "Confusion", "Frustration"]:
        if col in df_binary.columns:
            print(f"  {col}: {df_binary[col].value_counts().sort_index().to_dict()}")

    # Save binary labels
    output_path = os.path.join(PROCESSED_DIR, f"{obj}_labels_binary.csv")
    df_binary.to_csv(output_path, index=False)
    print(f"Saved binary labels to {output_path}")
    print(f"Binary {obj} dataset: {len(df_binary)} samples\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert multi-class labels to binary labels for DAiSEE dataset"
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["Train", "Test", "Validation"],
        help="Objects to process (default: Train Test Validation)",
    )
    args = parser.parse_args()

    print("Converting multi-class labels to binary labels...")
    print("Conversion rules: {0,1} → 0 (low), {2,3} → 1 (high)")
    print("- Boredom: 0->0, 1->0, 2->1, 3->1")
    print("- Engagement: 0->0, 1->0, 2->1, 3->1")
    print("- Confusion: 0->0, 1->0, 2->1, 3->1")
    print("- Frustration: 0->0, 1->0, 2->1, 3->1")
    print()

    for obj in tqdm(args.objects, desc="Processing objects"):
        process_object_labels(obj)

    print("Binary label conversion completed!")
    print("\nOutput files:")
    for obj in args.objects:
        output_path = os.path.join(PROCESSED_DIR, f"{obj}_labels_binary.csv")
        if os.path.exists(output_path):
            print(f"- {output_path}")
