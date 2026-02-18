# analyze_droplets.py

Detects and characterizes droplets in microscopy videos. This is the primary detection script that processes video frames to find laser-illuminated droplets (bright streaks), measures their geometry, and exports per-droplet data for downstream analysis.

## Dependencies

```bash
pip install opencv-python numpy matplotlib scipy
```

## Usage

### Test Mode (Recommended for First Use)

Edit the parameters in the `if __name__ == '__main__'` section at the bottom of the script:

```python
TEST_VIDEO = "your_video.mp4"
TEST_START_TIME = 5.0      # seconds
TEST_END_TIME = 10.0       # seconds
TEST_X_START = 100         # pixels
TEST_X_STOP = 500
TEST_Y_START = 50
TEST_Y_STOP = 400
TEST_THRESHOLD = 50
TEST_MIN_AREA = 10
TEST_MAX_AREA = 100
TEST_VIEW_FRAMES = True    # opens interactive viewer
```

Then run:

```bash
python analyze_droplets.py
```

### Command Line

```bash
python analyze_droplets.py VIDEO [OPTIONS]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `VIDEO` | Path to video file |
| `-t START END` or `--start-time`/`--end-time` | Time range in seconds |
| `-x START STOP` or `--x-start`/`--x-stop` | X pixel range |
| `-y START STOP` or `--y-start`/`--y-stop` | Y pixel range |

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--threshold` | 50 | Brightness threshold (0-255) |
| `--min-area` | 10 | Minimum droplet area in pixels |
| `--max-area` | 100 | Maximum droplet area in pixels |
| `--view-frames` | off | Open interactive viewer before analysis |
| `--plot-first-frame` | off | Display annotated first frame after analysis |
| `-o FILE` | none | Save results to JSON file |

**Examples:**

```bash
# Basic analysis with interactive viewer
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 --view-frames

# Adjust detection sensitivity and save results
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 \
    --threshold 80 --min-area 15 --max-area 150 -o results.json
```

## Interactive Viewer

Activated with `--view-frames` or `TEST_VIEW_FRAMES = True`.

### Controls

| Control | Action |
|---------|--------|
| `<<< 10`, `< 1`, `1 >`, `10 >>>` | Navigate frames |
| Jump to Frame | Enter frame number, press Enter |
| X/Y start/stop text boxes | Set exact ROI coordinates |
| Select ROI button | Click-and-drag ROI on image |
| Threshold / Min area / Max area | Tune detection parameters |
| Start (s) / End (s) | Adjust analysis time window |

### Visualization Modes

| Mode | Description |
|------|-------------|
| **Show Droplets** | Preview detection: green circles = boundaries, red lines = streak length, blue lines = streak width |
| **Show Sum View** | Persistence/burn-in view showing cumulative droplet positions. Toggle between "Analysis Only" and "Entire Video" |
| **Show Processed** | View the preprocessed binary image (CLAHE, blur, threshold, morphology) |
| **Run Analysis** | Accept current parameters and start full analysis |

## Detection Algorithm

1. **CLAHE** contrast enhancement (clipLimit=2.0, tileGridSize=8x8)
2. **Gaussian blur** (3x3 kernel) for noise reduction
3. **Binary thresholding** at user-specified brightness level
4. **Morphological closing** (3x3 kernel, 1 iteration) to connect nearby pixels
5. **Contour detection** (external contours only)
6. **Size filtering** by contour area (min/max)
7. **Minimum area rectangle** fitting for streak length, width, and angle

Each droplet is tagged with `in_roi` to indicate whether its centroid falls inside the spatial ROI.

## Output Files

### Console Output

Prints video properties, analysis region, frame progress, count statistics, streak statistics, and density.

### Annotated Frame Plot

First analyzed frame showing: red ROI rectangle, green circles (droplet boundaries), red lines (streak length), blue lines (streak width).

### Histogram Plots

Three histograms: droplet count distribution (with density twin axis), streak length distribution, and streak width distribution. Each shows mean and standard deviation.

### `droplet_data.npy`

Structured NumPy array with per-droplet records:

```python
import numpy as np
data = np.load('droplet_data.npy')

# Fields: frame, droplet_id, x, y, length, width, angle, in_roi,
#         length_start_x, length_start_y, length_end_x, length_end_y
```

### `test_results.json` (with `-o` flag)

Complete results including video path, analysis parameters, statistical summary, and per-frame droplet measurements.

## Parameter Tuning Guide

| Parameter | Default | Too High | Too Low |
|-----------|---------|----------|---------|
| Threshold | 50 | Misses dim droplets | Detects background noise |
| Min Area | 10 px | Misses small droplets | Includes noise |
| Max Area | 100 px | Counts clumps as one | Misses large droplets |

Use the **persistence view** to identify optimal ROI placement, and **Show Droplets** to verify detection quality.
