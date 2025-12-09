import os
import shutil
import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm

# --- Constants ---
ROOT_DIR = "../DAiSEE"
PROCESSED_DIR = os.path.join(ROOT_DIR, "Processed")
INPUT_DATA_DIR = os.path.join(PROCESSED_DIR, "squeeze-data")
INPUT_LABELS_FILE = os.path.join(PROCESSED_DIR, "squeeze-labels.csv")

OUTPUT_DATA_DIR = os.path.join(PROCESSED_DIR, "Augmented")
OUTPUT_LABELS_FILE = os.path.join(PROCESSED_DIR, "augmented-labels.csv")

# Augmentation Multipliers
MULTIPLIERS = {
    "Frustration": 20, # Class 1
    "Engagement": 16,  # Class 0
    "Confusion": 10,   # Class 1
    "Boredom": 3       # Class 1
}

def get_multiplier(row):
    mults = [0]
    if row['Frustration'] == 1: mults.append(MULTIPLIERS['Frustration'])
    if row['Engagement'] == 0: mults.append(MULTIPLIERS['Engagement'])
    if row['Confusion'] == 1: mults.append(MULTIPLIERS['Confusion'])
    if row['Boredom'] == 1: mults.append(MULTIPLIERS['Boredom'])
    return max(mults)

class VideoAugmenter:
    def __init__(self, width=224, height=224):
        self.width = width
        self.height = height

    def augment(self, frames):
        """
        Apply consistent transformations to a sequence of frames.
        frames: numpy array of shape (T, H, W, C) - uint8
        """
        # 1. Random Parameters (Consistent for the whole clip)
        do_flip = random.random() < 0.5
        
        # Color Jitter
        beta = random.uniform(-20, 20)      # Brightness
        alpha = random.uniform(0.8, 1.2)    # Contrast
        
        # Gaussian Blur
        do_blur = random.random() < 0.15
        
        # Random Crop
        scale = random.uniform(0.95, 1.0)
        ratio = random.uniform(0.95, 1.05)
        
        h_orig, w_orig = frames.shape[1], frames.shape[2]
        
        target_area = h_orig * w_orig * scale
        target_aspect_ratio = ratio
        
        try:
            w_crop = int(round(np.sqrt(target_area * target_aspect_ratio)))
            h_crop = int(round(np.sqrt(target_area / target_aspect_ratio)))
        except:
            w_crop, h_crop = w_orig, h_orig
            
        # Bound checking
        if w_crop > w_orig: w_crop = w_orig
        if h_crop > h_orig: h_crop = h_orig
        
        x_crop = random.randint(0, w_orig - w_crop)
        y_crop = random.randint(0, h_orig - h_crop)
        
        # Random Erasing
        do_erase = random.random() < 0.25
        if do_erase:
            erase_area = h_orig * w_orig * random.uniform(0.02, 0.05)
            erase_ratio = random.uniform(0.3, 3.3)
            
            w_erase = int(round(np.sqrt(erase_area * erase_ratio)))
            h_erase = int(round(np.sqrt(erase_area / erase_ratio)))
            
            # Bound
            if w_erase >= w_orig: w_erase = w_orig - 1
            if h_erase >= h_orig: h_erase = h_orig - 1
            
            x_erase = random.randint(0, w_orig - w_erase)
            y_erase = random.randint(0, h_orig - h_erase)
        
        # 2. Apply Transforms
        aug_frames = []
        for frame in frames:
            # frame is H,W,C
            
            # Horizontal Flip
            if do_flip:
                frame = cv2.flip(frame, 1)
            
            # Brightness / Contrast
            # F = alpha * I + beta
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            # Gaussian Blur
            if do_blur:
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                
            # Crop
            frame = frame[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
            
            # Resize back to original
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            
            # Random Erasing
            if do_erase:
                # Fill with random noise or constant gray
                # Using 127 gray for simplicity or random noise
                # noise = np.random.randint(0, 255, (h_erase, w_erase, 3), dtype=np.uint8)
                # frame[y_erase:y_erase+h_erase, x_erase:x_erase+w_erase] = noise
                frame[y_erase:y_erase+h_erase, x_erase:x_erase+w_erase] = 127
                
            aug_frames.append(frame)
            
        return np.array(aug_frames, dtype=np.uint8)

def augment_and_save(filename, row, augmenter, output_labels):
    src_path = os.path.join(INPUT_DATA_DIR, filename)
    if not os.path.exists(src_path):
        return

    try:
        # Load Data
        clip_np = np.load(src_path) # (T, 224, 224, 3)

        multiplier = get_multiplier(row)
        if multiplier <= 0:
            return

        base_name = os.path.splitext(filename)[0]
        
        for i in range(multiplier):
            # Apply consistent augmentation
            aug_np = augmenter.augment(clip_np)
            
            # Save
            new_filename = f"aug_{i}_{base_name}.npy"
            dst_path = os.path.join(OUTPUT_DATA_DIR, new_filename)
            
            # Ensure shape is correct (T, 224, 224, 3)
            # cv2.resize handles the spatial dims, T is preserved.
            
            np.save(dst_path, aug_np)
            
            # Record Label
            new_row = row.copy()
            new_row['ClipID'] = f"aug_{i}_{base_name}"
            output_labels.append(new_row)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
        print(f"Created {OUTPUT_DATA_DIR}")

    print("Loading labels...")
    df = pd.read_csv(INPUT_LABELS_FILE)
    print(f"Total samples: {len(df)}")

    augmenter = VideoAugmenter(224, 224)
    new_labels = []

    print("Starting augmentation...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        clip_id = str(row['ClipID'])
        # ClipID might be '123.avi'. We need '123.npy'.
        base_id = os.path.splitext(clip_id)[0]
        filename = f"{base_id}.npy"
            
        augment_and_save(filename, row, augmenter, new_labels)

    if new_labels:
        aug_df = pd.DataFrame(new_labels)
        aug_df.to_csv(OUTPUT_LABELS_FILE, index=False)
        print(f"Saved {len(aug_df)} augmented labels to {OUTPUT_LABELS_FILE}")
    else:
        print("No augmentations generated.")

if __name__ == "__main__":
    main()
