# Label Analysis Comparison

## Overview
This report compares the class distribution of the dataset **Before** and **After** data augmentation.

### Dataset Size
- **Original**: 8,986 samples
- **Augmented**: 26,228 samples (Added)
- **Total**: 35,214 samples

### Class Distribution (Presence of Label)

| Label | Original Count | Original % | Combined Count (After) | Combined % | Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Boredom** | 2,256 | 25.1% | 18,258 | **51.85%** | ⬆️ 26.7% |
| **Engagement** | 8,467 | 94.2% | 26,067 | **74.02%** | ⬇️ 20.2% (Better Balance) |
| **Confusion** | 845 | 9.4% | 11,443 | **32.50%** | ⬆️ 23.1% |
| **Frustration** | 427 | 4.8% | 8,967 | **25.46%** | ⬆️ 20.7% |

## Analysis
The augmentation strategy successfully addressed the class imbalance:

1.  **Minority Classes Boosted**:
    *   **Confusion** and **Frustration** saw a massive increase from <10% to ~25-32%. This will allow the model to learn these features much more effectively.
    *   **Boredom** increased from 25% to ~52%, achieving a near-perfect balance.

2.  **Majority Class Controlled**:
    *   **Engagement** (originally 94%) was effectively diluted to 74%. While still the majority, the presence of the negative class (0) has increased significantly due to the augmentation of non-engaged samples (Engagement=0 were targeted with 16x augmentation).

## Visual Check
Augmented samples were verified for visual integrity using `scripts/visualize_aug.py`. 
See `augmented_samples.jpg` for examples of the transforms (Color Jitter, Crop, Blur, Erase).
