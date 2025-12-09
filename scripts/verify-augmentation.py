import pandas as pd
import os
import glob
import numpy as np

# --- Constants ---
ROOT_DIR = "../DAiSEE"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Processed")
ORIGINAL_LABELS = os.path.join(PROCESSED_DIR, "squeeze-labels.csv")
AUGMENTED_LABELS = os.path.join(PROCESSED_DIR, "augmented-labels.csv")
AUGMENTED_DATA_DIR = os.path.join(PROCESSED_DIR, "Augmented")

def verify():
    print("--- Verification Stats ---")
    
    # 1. Check Files
    if not os.path.exists(ORIGINAL_LABELS):
        print(f"Error: Original labels not found at {ORIGINAL_LABELS}")
        return
        
    orig_df = pd.read_csv(ORIGINAL_LABELS)
    print(f"Original Samples: {len(orig_df)}")
    
    aug_df = pd.DataFrame()
    if os.path.exists(AUGMENTED_LABELS):
        aug_df = pd.read_csv(AUGMENTED_LABELS)
        print(f"Augmented Samples: {len(aug_df)}")
    else:
        print("Warning: No augmented label file found yet.")
        
    # 2. Merge
    if not aug_df.empty:
        merged_df = pd.concat([orig_df, aug_df], ignore_index=True)
    else:
        merged_df = orig_df
        
    print(f"Total Combined Samples: {len(merged_df)}")
    
    # 3. Distribution
    target_labels = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    
    print("\n--- New Label Distribution ---")
    data = []
    for label in target_labels:
        col = next((c for c in merged_df.columns if c.lower() == label.lower()), None)
        if col:
            counts = merged_df[col].value_counts()
            prop = merged_df[col].value_counts(normalize=True) * 100
            
            c0 = counts.get(0, 0)
            c1 = counts.get(1, 0)
            p1 = prop.get(1, 0.0)
            
            data.append({
                "Label": label,
                "Class 0": c0,
                "Class 1": c1,
                "% Present": f"{p1:.2f}%"
            })
            
    res_df = pd.DataFrame(data)
    print(res_df.to_string(index=False))
    
    # 4. Check a few files
    print("\n--- File Integrity Check ---")
    if not aug_df.empty:
        sample_aug = aug_df.sample(n=min(3, len(aug_df)))
        for idx, row in sample_aug.iterrows():
            clip_id = row['ClipID'] 
            # Check if filename needs .npy (assuming saved as .npy)
            if not str(clip_id).endswith('.npy'):
                filename = f"{clip_id}.npy"
            else:
                filename = clip_id
                
            path = os.path.join(AUGMENTED_DATA_DIR, filename)
            if os.path.exists(path):
                try:
                    arr = np.load(path)
                    print(f"[OK] {filename}: {arr.shape}, {arr.dtype}")
                except Exception as e:
                    print(f"[FAIL] {filename}: {e}")
            else:
                print(f"[MISSING] {filename}")
    else:
        print("No augmented files to check.")

if __name__ == "__main__":
    verify()
