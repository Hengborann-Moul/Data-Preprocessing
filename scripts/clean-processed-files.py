import os

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

ROOT_DIR = "../DAiSEE"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Processed")


def get_npy_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".npy")]


def get_labels_df(obj) -> DataFrame:
    return pd.read_csv(os.path.join(PROCESSED_DIR, f"{obj}_labels.csv"))


if __name__ == "__main__":
    objs = ["Train", "Test", "Validation"]

    # Handle No Labels files, create No_Labels Folder if not exists
    no_labels_dir = os.path.join(ROOT_DIR, "No_Labels")
    if not os.path.exists(no_labels_dir):
        os.makedirs(no_labels_dir)

    for obj in tqdm(objs, desc="Processing datasets: "):
        labels_df = get_labels_df(obj)
        npy_files = get_npy_files(os.path.join(PROCESSED_DIR, obj))

        if len(labels_df) != len(npy_files):
            print(
                f"Mismatch in {obj}: Labels - {len(labels_df)}, NPY files - {len(npy_files)}"
            )
            # Identify and move files without labels
            labeled_files = set(
                labels_df["ClipID"].apply(lambda x: f"{os.path.splitext(x)[0]}.npy")
            )
            for npy_file in npy_files:
                if npy_file not in labeled_files:
                    os.rename(
                        os.path.join(PROCESSED_DIR, obj, npy_file),
                        os.path.join(no_labels_dir, npy_file),
                    )
                    print(f"Moved {npy_file} to No_Labels directory.")
        else:
            print(f"All files in {obj} have corresponding labels.")
