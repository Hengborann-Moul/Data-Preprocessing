import os

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

ROOT_DIR = "../DAiSEE"


def get_preprocessed_files(obj):
    processed_dir = os.path.join(ROOT_DIR, "Processed", obj)
    if not os.path.exists(processed_dir):
        return set()
    return {
        os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith(".npy")
    }


def get_labels_from_csv(idx, obj) -> DataFrame:
    csv_path = os.path.join(ROOT_DIR, f"Labels/{obj}Labels.csv")
    df = pd.read_csv(csv_path)
    # The ClipID column contains values like "{idx}.avi" or "{idx}.mp4"
    # Check if any ClipID matches "{idx}.avi" or "{idx}.mp4"
    clipid_avi = f"{idx}.avi"
    clipid_mp4 = f"{idx}.mp4"
    if not ((df["ClipID"] == clipid_avi).any() or (df["ClipID"] == clipid_mp4).any()):
        # Try all csv path
        all_csv_paths = os.path.join(ROOT_DIR, "Labels/AllLabels.csv")
        df_all = pd.read_csv(all_csv_paths)
        if (df_all["ClipID"] == clipid_avi).any() or (
            df_all["ClipID"] == clipid_mp4
        ).any():
            return df_all[
                (df_all["ClipID"] == clipid_avi) | (df_all["ClipID"] == clipid_mp4)
            ]
        raise ValueError(f"Index {idx} not found in {csv_path}")
    return df[(df["ClipID"] == clipid_avi) | (df["ClipID"] == clipid_mp4)]


if __name__ == "__main__":
    objs = ["Train", "Test", "Validation"]
    for obj in tqdm(objs, desc="Checking preprocessed datasets: "):
        processed_files = get_preprocessed_files(obj)
        print(f"{obj} dataset has {len(processed_files)} preprocessed files.")

        # Create an Empty DataFrame to hold labels
        df_labels = pd.DataFrame()
        for file in tqdm(list(processed_files), desc=f"Processing {obj} files: "):
            try:
                file_labels = get_labels_from_csv(file, obj)
                df_labels = pd.concat([df_labels, file_labels], ignore_index=True)
            except ValueError as e:
                print(e)
                continue
        print(f"Collected labels for {len(df_labels)} files in {obj} dataset.")
        # Save the DataFrame to a CSV file
        output_csv_path = os.path.join(ROOT_DIR, "Processed", f"{obj}_labels.csv")
        df_labels.to_csv(output_csv_path, index=False)
        print(f"Saved labels to {output_csv_path}")
        print("\n")
