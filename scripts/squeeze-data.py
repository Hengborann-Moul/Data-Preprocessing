import os
import shutil
import pandas as pd
from tqdm import tqdm

ROOT_DIR = "../DAiSEE"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Processed")
OUTPUT_DATA_DIR = os.path.join(PROCESSED_DIR, "squeeze-data")
OUTPUT_LABELS_FILE = os.path.join(PROCESSED_DIR, "squeeze-labels.csv")

# Map folder names to label prefixes if necessary, or just lists
DATA_FOLDERS = ["Train1", "Test1", "Validation1"]
LABEL_FILES = {
    "Train1": "Train_labels_binary.csv",
    "Test1": "Test_labels_binary.csv",
    "Validation1": "Validation_labels_binary.csv"
}

def merge_data():
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
        print(f"Created directory: {OUTPUT_DATA_DIR}")
    else:
        print(f"Directory already exists: {OUTPUT_DATA_DIR}")

    print("Merging data files...")
    for folder in DATA_FOLDERS:
        source_folder = os.path.join(PROCESSED_DIR, folder)
        if not os.path.exists(source_folder):
            print(f"Warning: Source folder not found: {source_folder}")
            continue
        
        files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]
        print(f"Processing {folder} ({len(files)} files)...")
        
        for filename in tqdm(files, desc=f"Copying from {folder}"):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(OUTPUT_DATA_DIR, filename)
            
            # Check if file exists to avoid unnecessary writes/errors
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

def merge_labels():
    print("\nMerging label files...")
    dfs = []
    
    for folder in DATA_FOLDERS:
        label_file = LABEL_FILES.get(folder)
        if not label_file:
            continue
            
        label_path = os.path.join(PROCESSED_DIR, label_file)
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}")
            continue
            
        print(f"Reading {label_file}...")
        df = pd.read_csv(label_path)
        dfs.append(df)
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        # Verify no duplicates based on ClipID if needed, but for now just merge
        
        print(f"Total merged records: {len(merged_df)}")
        merged_df.to_csv(OUTPUT_LABELS_FILE, index=False)
        print(f"Saved merged labels to {OUTPUT_LABELS_FILE}")
    else:
        print("No label data found to merge.")

if __name__ == "__main__":
    merge_data()
    merge_labels()
    print("\nSqueeze operation completed!")
