# Droplet Analysis Pipeline for Microscopy Videos

A suite of tools for detecting, tracking, and statistically analyzing droplets in laser-illuminated microscopy videos. Droplets appear as bright streaks whose length, width, and motion encode physical properties such as speed, size, and temperature.

## Pipeline Overview

```
Video File
    |
    v
analyze_droplets.py    --> droplet_data.npy, test_results.json
    |
    v
track_droplets.py      --> linked tracks (endpoint vectors, streamlines)
    |
    v
analyze_distributions.py  --> count/width/speed histograms with physical fits
```

**Supporting utility:**
- `set_distance_threshold.py` -- automatically determines the optimal linking distance for tracking

## Scripts

| Script | Purpose | Documentation |
|--------|---------|---------------|
| [analyze_droplets.py](analyze_droplets.py) | Detect droplets and measure streak geometry | [analyze_droplets.md](analyze_droplets.md) |
| [track_droplets.py](track_droplets.py) | Link detections across frames into tracks | [track_droplets.md](track_droplets.md) |
| [analyze_distributions.py](analyze_distributions.py) | Fit statistical distributions to droplet properties | [analyze_distributions.md](analyze_distributions.md) |
| [set_distance_threshold.py](set_distance_threshold.py) | Find optimal tracking distance threshold | [set_distance_threshold.md](set_distance_threshold.md) |

## Installation

```bash
pip install opencv-python numpy matplotlib scipy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Run detection with the interactive viewer to tune parameters:
   ```bash
   python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 --view-frames
   ```

2. Link detections into tracks:
   ```bash
   python track_droplets.py
   ```

3. Analyze distributions:
   ```bash
   python analyze_distributions.py
   ```

See each script's `.md` file for detailed usage, parameters, and output descriptions.

## Data Flow

### `droplet_data.npy`

Structured NumPy array produced by `analyze_droplets.py` and consumed by the downstream scripts. Per-droplet fields: `frame`, `droplet_id`, `x`, `y`, `length`, `width`, `angle`, `in_roi`.

### `test_results.json`

Metadata and summary statistics from `analyze_droplets.py`, including calibration values (`pixels_per_um`, `fps`, `exposure_time_s`) used by `analyze_distributions.py`.

## Workflow Recommendations

1. **Initial setup** -- Use the interactive viewer in `analyze_droplets.py` to select ROI and tune detection parameters. The persistence view helps identify the illuminated region.
2. **Quality check** -- Verify detections visually (green circles, red/blue axis lines) before running full analysis.
3. **Tracking** -- Start with the automatic threshold selection (`max_distance=None`) in `track_droplets.py`, then inspect streamlines.
4. **Batch processing** -- Once parameters are tuned, reuse them across videos with consistent illumination conditions.

## License

[Specify your license here]
