import pandas as pd
import os

# Path found via find command
LABELS_FILE = "/home/tadashi/Documents/MS-DAS/DAiSEE/Processed/squeeze-labels.csv"

def analyze_labels():
    if not os.path.exists(LABELS_FILE):
        print(f"Error: File not found at {LABELS_FILE}")
        return

    print(f"Reading {LABELS_FILE}...")
    try:
        df = pd.read_csv(LABELS_FILE)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print(f"Columns found: {list(df.columns)}")
    
    # Target labels based on user request (handling potential case differences)
    target_labels = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    
    print("\n--- Label Distribution ---")
    for label in target_labels:
        # Find column matching the label (case insensitive)
        col = next((c for c in df.columns if c.lower() == label.lower()), None)
        
        if col:
            print(f"\nDistribution for '{col}':")
            print(df[col].value_counts())
            print(f"\nPercentage for '{col}':")
            print(df[col].value_counts(normalize=True) * 100)
        else:
            print(f"\nWarning: Column for '{label}' not found.")

if __name__ == "__main__":
    analyze_labels()
