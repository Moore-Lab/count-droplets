# Droplet Analysis Pipeline for Microscopy Videos

A suite of tools for detecting, tracking, statistically analyzing, and performing longitudinal time-series analysis of droplets in laser-illuminated microscopy videos.  Droplets appear as bright streaks whose length, width, and motion encode physical properties such as speed, size, and temperature.

## Pipeline Overview

```
Video File
    |
    v
analyze_droplets.py   --> data/<basename>/<basename>_droplet_data.npy
                          data/<basename>/<basename>_test_results.json
                          logs/<basename>_<timestamp>.json
    |
    v
track_droplets.py     --> linked tracks (endpoint vectors, streamlines)
    |
    v
analyze_distributions.py  --> count/width/speed histograms with physical fits
    |
    v
longitudinal_analysis.py  --> data/<basename>/longitudinal_analysis_<basename>_<timestamp>.npy
                              plots/<basename>/chunk_XXXX.png
                              plots/<basename>/<basename>_longitudinal_<param>.png
    |
    v
environmental_correlations.py --> plots/environmental_correlations/
```

**Supporting utility:**
- `set_distance_threshold.py` -- automatically determines the optimal linking distance for tracking

## Scripts

| Script | Purpose | Documentation |
|--------|---------|---------------|
| [analyze_droplets.py](analyze_droplets.py) | Detect droplets, measure streak geometry, save log | [analyze_droplets.md](analyze_droplets.md) |
| [track_droplets.py](track_droplets.py) | Link detections across frames into tracks | [track_droplets.md](track_droplets.md) |
| [analyze_distributions.py](analyze_distributions.py) | Fit statistical distributions to droplet properties | [analyze_distributions.md](analyze_distributions.md) |
| [set_distance_threshold.py](set_distance_threshold.py) | Find optimal tracking distance threshold | [set_distance_threshold.md](set_distance_threshold.md) |
| [longitudinal_analysis.py](longitudinal_analysis.py) | Chunk-based longitudinal time-series analysis | [longitudinal_analysis.md](longitudinal_analysis.md) |
| [environmental_correlations.py](environmental_correlations.py) | Correlate droplet statistics with environmental sensor data | — |
| [query_db.py](query_db.py) | Query InfluxDB for cryostat/gas-handling sensor data | — |

## Installation

```bash
pip install opencv-python numpy matplotlib scipy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Detect droplets and tune parameters

Use the interactive GUI to select ROI and tune detection parameters:

```bash
python analyze_droplets.py video.mp4 -t 5 30 -x 100 500 -y 50 400 --view-frames
```

The interactive viewer provides:
- **Frame navigation** — step through frames one at a time or in jumps of 10
- **ROI selection** — click-and-drag or type pixel coordinates
- **Detection preview** — green circles around detected droplets
- **Sum (persistence) view** — accumulated image showing where droplets travel
- **Processed view** — binary image after CLAHE/blur/threshold/morphology

#### New GUI controls

| Control | Description |
|---------|-------------|
| **FPS ovr:** textbox | Override the video's native frame rate for time calculations.  Leave blank to use the video fps. |
| **Exp (s):** textbox | Camera exposure time in seconds.  Used to compute velocity from streak length (`v = length / exposure`).  Leave blank to use `1/fps`. |
| **Load Log** button | Open a file dialog to load a `logs/*.json` file and auto-populate all analysis parameters. |

### 2. Run analysis and save log

After closing the viewer, the script:
1. Runs droplet detection on the selected time window
2. Saves `data/<basename>/<basename>_droplet_data.npy`
3. Saves `data/<basename>/<basename>_test_results.json`
4. Saves `logs/<basename>_<YYYY-MM-DD_HH-MM-SS>.json` — **the analysis log**

### 3. Link detections into tracks

Edit `VIDEO_BASENAME` at the bottom of `track_droplets.py`, then run:

```bash
python track_droplets.py
```

### 4. Analyze distributions

Edit `VIDEO_BASENAME` at the bottom of `analyze_distributions.py`, then run:

```bash
python analyze_distributions.py
```

### 5. Longitudinal analysis

Edit the **USER SETTINGS** block in `longitudinal_analysis.py`, then run:

```bash
python longitudinal_analysis.py
```

This partitions the detected data into time or frame chunks and runs `analyze_distributions` on each chunk.  See [longitudinal_analysis.md](longitudinal_analysis.md) for full documentation.

### 6. Environmental correlations

Edit `LONGITUDINAL_FILE` in `environmental_correlations.py`, then run:

```bash
python environmental_correlations.py
```

This queries the InfluxDB database for temperature (RTDs) and pressure (Setra 225 gauges) over the same time window and produces scatter plots in `plots/environmental_correlations/`.

## Data Directory Structure

```
count-droplets/
├── data/
│   └── <video_basename>/
│       ├── <basename>_droplet_data.npy       # per-droplet detections
│       ├── <basename>_tracking_data.npy      # per-droplet with endpoint coordinates
│       ├── <basename>_test_results.json      # analysis metadata and statistics
│       └── longitudinal_analysis_<basename>_<timestamp>.npy
├── logs/
│   └── <basename>_<YYYY-MM-DD_HH-MM-SS>.json  # analysis parameter log
├── plots/
│   ├── <video_basename>/                       # chunk plots + longitudinal plots
│   └── environmental_correlations/             # env correlation scatter plots
└── videos/
```

## Data Flow

### `data/<basename>/<basename>_droplet_data.npy`

Structured NumPy array produced by `analyze_droplets.py` and consumed by downstream scripts. Per-droplet fields: `frame`, `droplet_id`, `x`, `y`, `length`, `width`, `angle`.

### `data/<basename>/<basename>_test_results.json`

Metadata and summary statistics from `analyze_droplets.py`, including calibration values (`pixels_per_um`, `fps`, `exposure_time_s`) used by `analyze_distributions.py` and `longitudinal_analysis.py`.

### `logs/<basename>_<timestamp>.json`

Parameter log saved automatically after each analysis run.  Contains all detection and ROI parameters, fps override, exposure time, and paths to the data files.  Load this into `longitudinal_analysis.py` or use the **Load Log** button in the GUI to reproduce or continue an analysis.

### `data/<basename>/longitudinal_analysis_<basename>_<timestamp>.npy`

Structured NumPy array with one row per chunk.  Fields include counts, density, width, speed, projected velocities, wind direction, and the UTC start timestamp.  See [longitudinal_analysis.md](longitudinal_analysis.md) for the full field list.

## Workflow Recommendations

1. **Initial setup** — Use the interactive viewer in `analyze_droplets.py` to select ROI and tune detection parameters.  The persistence view helps identify the illuminated region.
2. **Parameter logging** — After a good analysis, the log file in `logs/` records all settings so you can reproduce the run or load it for longitudinal analysis.
3. **Longitudinal analysis** — Point `longitudinal_analysis.py` at the log file and choose a chunk size.  Both frame-count and time-based chunking are supported.
4. **Environmental correlations** — Once you have a longitudinal numpy file with a `start_utc` set, run `environmental_correlations.py` to overlay sensor data.
5. **Batch processing** — Once parameters are tuned, reuse the log file across sessions or load it into the GUI with the **Load Log** button.

## License

[Specify your license here]
