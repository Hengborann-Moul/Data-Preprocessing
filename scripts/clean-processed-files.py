import argparse
import os

import numpy as np
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Clean processed files and validate frame counts"
    )
    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Expected number of frames in each .npy file",
    )
    args = parser.parse_args()

    frame_count = args.frame
    objs = ["Train", "Test", "Validation"]

    # Handle No Labels files, create No_Labels Folder if not exists
    no_labels_dir = os.path.join(ROOT_DIR, "No_Labels")
    if not os.path.exists(no_labels_dir):
        os.makedirs(no_labels_dir)

    for obj in tqdm(objs, desc="Processing datasets: "):
        labels_df = get_labels_df(obj)
        npy_files = get_npy_files(os.path.join(PROCESSED_DIR, f"{obj}"))

        # if len(labels_df) != len(npy_files):
        # print(
        #     f"Mismatch in {obj}: Labels - {len(labels_df)}, NPY files - {len(npy_files)}"
        # )
        # Identify and move files without labels
        labeled_files = set(
            labels_df["ClipID"].apply(lambda x: f"{os.path.splitext(x)[0]}.npy")
        )
        for npy_file in npy_files:
            file_path = os.path.join(PROCESSED_DIR, f"{obj}", npy_file)
            should_move = False
            move_reason = ""

            # Check frame count
            try:
                npy_data = np.load(file_path)
                if len(npy_data) != frame_count:
                    should_move = True
                    move_reason = f"incorrect frame count (expected {frame_count}, got {len(npy_data)})"

                # Check if the file has label
                if npy_file not in labeled_files:
                    should_move = True
                    move_reason = "no corresponding label"
            except Exception as e:
                should_move = True
                move_reason = f"error loading file: {str(e)}"

            if should_move:
                os.rename(file_path, os.path.join(no_labels_dir, npy_file))
                print(f"Moved {npy_file} to No_Labels directory ({move_reason}).")
        # else:
        #     print(f"All files in {obj} have corresponding labels.")
