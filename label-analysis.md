# Label Analysis Comparison

## Overview
This report compares the class distribution of the dataset **Before** and **After** data augmentation (Strategy: Fine-Tuned Combination-Based Sampling).

### Dataset Size
- **Original**: 8,986 samples
- **Augmented**: 6,593 samples (Added)
- **Total**: 15,579 samples

### Class Distribution (Presence of Label)

| Label | Original Count | Original % | Combined Count (After) | Combined % | Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Boredom** | 2,256 | 25.1% | 6,099 | **39.1%** | ⬆️ 14.0% |
| **Engagement** | 8,467 | 94.2% | 11,731 | **75.3%** | ⬇️ 18.9% (Better Balance) |
| **Confusion** | 845 | 9.4% | 4,501 | **28.9%** | ⬆️ 19.5% |
| **Frustration** | 427 | 4.8% | 4,138 | **26.6%** | ⬆️ 21.8% |

## Analysis
The refined augmentation strategy successfully addressed the class imbalance while minimizing the number of synthetic samples:

1.  **Efficient Balancing**: 
    *   We added only **6,593** samples (vs 18,233 in the previous attempt), yet achieved similar balance improvements.
    *   **Confusion** and **Frustration** are now robustly represented (~26-29%).
    
2.  **Targeted Combinations**:
    *   Rare label combinations (e.g., *Frustrated but NOT Engaged*) were boosted 20x to reach ~600 samples each.
    *   Majority combinations (e.g., *Engaged only*) were untouched, preserving the natural distribution where possible.

## Visual Check
Augmented samples were verified for visual integrity using `scripts/visualize_aug.py`. 
See `augmented_samples.jpg` for examples of the transforms (Color Jitter, Crop, Blur, Erase).
