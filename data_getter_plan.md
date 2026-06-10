# Plan for `DataGetter` Class

## Objective
Bridge the gap between raw filesystem discovery and the input requirements of a Neural Network (NN), serving as the orchestration layer for `ImageProcessor`.

## Core Responsibilities
1. **Discovery**: Crawl directory trees to find valid source files (`.pdf`, `.png`, `.jpg`).
2. **Orchestration**: Manage instances of `ImageProcessor` to convert raw files into normalized tensor-ready arrays.
3. **Data Normalization**: Flatten the hierarchical output from `ImageProcessor.get_cell_images()` into a linear stream for training/inference.
4. **Metadata Mapping**: Maintain associations between processed cells and their original source file/location.

## Proposed Class Interface

### 1. Initialization
- `__init__(self, root_dir: str, extensions=('.pdf', '.png', '.jpg'))`
  - Scans `root_dir` for files matching `extensions`.
  - Stores a manifest of available file paths.

### 2. The Generator (The Engine)
- `iter_batches(self, batch_size: int = 32)`
  - **Process**:
    1. Pick $N$ files from the manifest.
    2. For each file:
       - Instantiate `ImageProcessor(path)`.
       - Call `processor.get_cell_images()`.
       - Flatten dictionary `{row: [img, id]}` into a flat list of $(Image, Metadata)$ tuples.
    3. Accumulate up to `batch_size`.
    4. Convert numpy arrays to the desired format (e.g., float32 tensors).
    5. **Yield** batch of images and metadata.

### 3. Optimization Strategies (Advanced)
- **Pre-computation Mode**: A method to run processing once and save results as `.npy` or `.png` files in a `processed/` directory, avoiding redundant CV2 work during NN training.
- **Lazy Loading**: Only instantiate `ImageProcessor` at the moment of access within the generator.

## Implementation Workflow
1. [ ] Define `__init__` with file discovery using `glob` or `os.walk`.
2. [ ] Implement `iter_batches` logic with error handling for corrupted images.
3. [ ] Add support for returning labels/metadata alongside image batches.
4. [ ] (Optional) Implementation of the "Pre-computation" caching mechanism.
