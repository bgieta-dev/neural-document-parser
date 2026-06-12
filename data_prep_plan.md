# 🛠️ Data Preparation Plan: Handwritten Digit Recognition (Optimized)

## Objective
Leverage existing pre-processed binary data ($128 \times 128$, centered) and apply high-variance augmentations to maximize CNN generalization.

---

## 1. Ingestion & Normalization
*Since data is already centered/resized:*
*   **Loading:** Load binary images as `float32` tensors.
*   **Normalization:** Scale pixel values from $[0, 255] \to [0, 1]$ or $[-1, 1]$. This is critical for stable gradient descent.

## 2. Morphological Augmentation (The "Ink" Layer)
*Manipulate the binary shapes directly to simulate different writing styles:*
*   **Thickness Variation:** Randomly apply `cv2.dilate` and `cv2.erode` with variable kernel sizes ($2 \times 2$ to $5 \times 5$). This simulates varying pen pressure or ink thickness.
*   **Edge Smoothing:** Apply a small Gaussian blur followed by re-thresholding to create "softer" binary edges, simulating scanning/sensor artifacts.

## 3. Geometric Augmentation (The "Style" Layer)
*Introduce spatial variance:*
*   **Rotation:** Random rotation within $\pm 15^\circ$ to handle handwriting slant.
*   **Translation:** Shift the digit $x/y$ by up to $10\%$ of frame width/height to simulate imperfect localization.
*   **Elastic Deformation:** Use `scipy.ndimage.elastic_transform` to create non-linear distortions, simulating the natural "flow" and varying curves of human handwriting.

## 4. Synthetic Data Synthesis (The "Volume" Layer)
*To overcome the small sample size (1,000 samples):*
*   **Source:** Use EMNIST (standard digits).
*   **Transformation Pipeline:**
    1. Convert EMNIST to binary format.
    2. Apply heavy `cv2.dilate` to match your custom data's thickness profile.
    3. Add "Salt & Pepper" noise and re-threshold.
*   **Target:** Generate a synthetic dataset $5\times$ larger than the custom dataset to provide a robust baseline.

## 5. Dataset Composition Strategy
*   **Training Set:** Mix of **Augmented Real Samples** ($40\%$) and **Transformed Synthetic Data** ($60\%$).
*   **Validation/Test Set:** Use **Real Custom Samples only**. Never include synthetic data in the validation set to prevent "leaked" performance metrics.

---

## Implementation Order
1.  `ingest_and_normalize.py`: Handles loading and range scaling.
2.  `augment_engine.py`: Implements morphological, geometric, and elastic transforms.
3.  `dataset_builder.py`: Orchestrates the mix of real (augmented) and synthetic data into final training tensors.
