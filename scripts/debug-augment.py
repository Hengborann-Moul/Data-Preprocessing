import pandas as pd
import os

LABELS_FILE = "../DAiSEE/Processed/squeeze-labels.csv"
DATA_DIR = "../DAiSEE/Processed/squeeze-data"

if os.path.exists(LABELS_FILE):
    df = pd.read_csv(LABELS_FILE)
    print("Columns:", df.columns)
    print("First 5 rows:")
    print(df.head())
    
    print("\nCheck dtypes:")
    print(df.dtypes)
    
    # Check if files exist for first few
    print("\nFile check:")
    for i, row in df.head().iterrows():
        clip_id = str(row['ClipID'])
        if not clip_id.endswith('.npy'):
            fname = clip_id + ".npy"
        else:
            fname = clip_id
        path = os.path.join(DATA_DIR, fname)
        print(f"ID: {clip_id}, Path: {path}, Exists: {os.path.exists(path)}")
        
        # Check multiplier logic manually
        # Expected Logic:
        # Frustration=1 -> 20
        # Engagement=0 -> 16
        # Confusion=1 -> 10
        # Boredom=1 -> 3
        mults = [0]
        if row['Frustration'] == 1: mults.append(20)
        if row['Engagement'] == 0: mults.append(16)
        if row['Confusion'] == 1: mults.append(10)
        if row['Boredom'] == 1: mults.append(3)
        print(f"  -> Calculated Multiplier: {max(mults)}")
else:
    print("Labels file not found.")
