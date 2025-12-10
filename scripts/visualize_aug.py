import numpy as np
import cv2
import os
import random


AUGMENTED_DIR = "../DAiSEE/Processed/Augmented"
OUTPUT_IMAGE = "augmented_samples.jpg"


def visualize():
    files = [f for f in os.listdir(AUGMENTED_DIR) if f.endswith(".npy")]
    if not files:
        print("No augmented files found.")
        return

    selected = random.sample(files, 5)

    # Create a grid image: 5 columns (samples), 3 rows (frames 0, 5, 9)
    rows = []

    for fname in selected:
        path = os.path.join(AUGMENTED_DIR, fname)
        frames = np.load(path)  # T, H, W, C

        # Select frames 0, 5, 9
        idxs = [0, 5, 9]
        sample_frames = []
        for i in idxs:
            if i < len(frames):
                # RGB to BGR for cv2 saving if needed, but matplotlib needs RGB
                # frames are likely RGB if loaded from squeeze-data (check source)
                # But squeeze-data was direct copy. Assuming RGB.
                sample_frames.append(frames[i])
            else:
                sample_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Concatenate vertically for this sample
        col = np.vstack(sample_frames)
        rows.append(col)

    # Concatenate horizontally
    grid = np.hstack(rows)

    # Save as image using cv2 (Expects BGR)
    # If data is RGB, convert to BGR
    # Usually video frames are BGR in cv2, but if we process as RGB in augmenter...
    # Let's assume RGB for display and convert.
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_IMAGE, grid_bgr)
    print(f"Saved visualization to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    visualize()
