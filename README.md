# Data Preprocessing: Video → Frame Tensor (.npy)

## Objective

Build a consistent, model-ready frame dataset from raw videos. This tool:

- Samples frames at a target FPS
- Detects faces and crops a 224×224 region centered on the face; if no face is found, it performs a random 224×224 crop
- Packs all cropped frames into a single NumPy array and saves it as a .npy file

Typical use cases include preparing inputs for action recognition, facial analysis, or any vision model that expects fixed-size frame tensors.

## What it does ?

- Input: a video file (e.g., `videos/1100081044.avi`)
- Processing: sample frames at `target_fps` (default 3), crop to 224×224 using Haar-cascade face detection when possible, else random crop
- Output: a single `.npy` file like `video_data/video_frames.npy` with shape `(N, 224, 224, 3)` where `N` is the number of frames extracted

Notes:

- If fewer than 30 frames are produced, the script prints a warning (no padding is applied by default).
- The Haar cascade file `haarcascade_frontalface_default.xml` must be available in the working directory when you run the script (it’s included in this repo).

## Requirements

- Python >= 3.12
- Packages (declared in `pyproject.toml`):
  - numpy
  - opencv-python
  - pandas
  - plotly
  - ipykernel
  - nbformat

## Setup

You can set this project up using either uv (recommended if you already use it) or a standard venv + pip flow.

### Option A: Using uv (recommended)

1. Install uv (one-time; pick one):

- Using pipx: `pipx install uv`
- Using pip: `pip install uv`

2. From the repository root, create and sync the environment:

- `uv sync`

This creates a virtual environment and installs all dependencies pinned by `pyproject.toml`/`uv.lock`.

3. (Optional) Activate the environment shell for ad‑hoc commands:

- `uv run python --version`

### Option B: Using venv + pip

1. Create a virtual environment:

- `python3 -m venv .venv`

2. Activate it (macOS/Linux):

- `source .venv/bin/activate`

3. Install dependencies from `pyproject.toml`:

- `pip install -r <(python -c 'import tomllib,sys;print("\n".join(tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))')`

If your shell doesn’t support process substitution, you can manually install the packages listed above.

## How to run

The main entry point is `main.py`, which exposes `process_video_to_npy` and a simple script runner.

Important path note (macOS/Linux): the default `video_file` in `main.py` currently uses a Windows-style backslash path (`".\\videos\\1100081044.avi"`). Update it to a POSIX path before running, e.g.:

- `video_file = "videos/1100081044.avi"`

Then run one of the following from the repo root:

- With uv: `uv run python main.py`
- With venv: `python main.py`

You can also import and call the function directly for custom paths/outputs:

```
from main import process_video_to_npy
process_video_to_npy(
    video_path="path/to/your/video.mp4",
    output_dir="video_data",
    output_filename="my_frames.npy",
    target_fps=3,
)
```

### Output

- Saved at: `video_data/video_frames.npy` (configurable)
- Shape: `(N, 224, 224, 3)` in BGR color order (as produced by OpenCV)

## Notebooks

- `notebooks/data-overview-101.ipynb`: starter notebook for exploring outputs. After installing the environment, you can open it in VS Code or Jupyter.

If you need a kernel:

- With uv: `uv run python -m ipykernel install --user --name data-preprocessing`

## Project structure

```
.
├── main.py                               # Video → cropped frame tensor (.npy)
├── haarcascade_frontalface_default.xml   # Face detector used for centered crops
├── videos/                               # Sample/input videos (example: 1100081044.avi)
├── video_data/                           # Output directory for .npy arrays
├── notebooks/                            # Data exploration notebooks
├── pyproject.toml                        # Dependencies and Python version
├── uv.lock                               # Locked dependency versions (uv)
└── README.md
```

## Troubleshooting

- Error: Could not open video file…
  - Verify the file path and that you run the script from the repository root.
- Error: Could not load the Haar Cascade classifier.
  - Ensure `haarcascade_frontalface_default.xml` exists in the working directory.
- Frame too small for a 224×224 crop.
  - The script skips frames smaller than 224×224; ensure your source video has sufficient resolution.
- Output has fewer than 30 frames.
  - The script warns but does not pad. If you need fixed-length tensors, consider padding after loading the `.npy`.

## Next steps (ideas)

- Add CLI arguments to `main.py` (paths, fps, min frames, padding mode)
- Convert BGR→RGB before saving if your models expect RGB
- Batched processing for multiple videos and CSV/JSON manifests

---

If you run into any issues setting this up on macOS, share the exact error message and we’ll help you resolve it quickly.
