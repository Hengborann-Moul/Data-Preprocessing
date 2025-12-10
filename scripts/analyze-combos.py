import pandas as pd


def analyze_combinations(file_path, name):
    try:
        df = pd.read_csv(file_path)
        # Create a combination key
        df["combo"] = df.apply(
            lambda row: (
                int(row["Boredom"]),
                int(row["Engagement"]),
                int(row["Confusion"]),
                int(row["Frustration"]),
            ),
            axis=1,
        )
        counts = df["combo"].value_counts().sort_index()
        print(f"\n--- {name} Combinations ---")
        print(f"Total: {len(df)}")
        for combo, count in counts.items():
            print(f"{combo}: {count}")
        return counts
    except FileNotFoundError:
        print(f"\n{name} file not found.")
        return None


print("Analyzing distributions...")
orig_counts = analyze_combinations("../DAiSEE/Processed/squeeze-labels.csv", "Original")
aug_counts = analyze_combinations(
    "../DAiSEE/Processed/augmented-labels.csv", "Current Augmented"
)
