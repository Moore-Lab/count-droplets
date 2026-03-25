# longitudinal_analysis.py

Performs time-resolved (longitudinal) analysis of droplet data from a single video by partitioning the detection results into consecutive time or frame chunks and extracting summary statistics from each.

## Overview

```
logs/<basename>_<timestamp>.json   (log from analyze_droplets)
          |
          v
longitudinal_analysis.py
          |
          +---> data/<basename>/longitudinal_analysis_<basename>_<timestamp>.npy
          |
          +---> plots/<basename>/chunk_XXXX.png   (one per chunk)
          |
          +---> plots/<basename>/<basename>_longitudinal_<param>.png
```

## Prerequisites

Run `analyze_droplets.py` first (with the interactive GUI or `__main__` block). This produces:
- `data/<basename>/<basename>_droplet_data.npy`  — per-droplet structured array
- `data/<basename>/<basename>_test_results.json` — analysis metadata
- `logs/<basename>_<timestamp>.json`             — analysis log

## Usage

Edit the **USER SETTINGS** block at the bottom of `longitudinal_analysis.py`:

```python
LOG_PATH          = "logs/xenon_2026-03-11_09-00-00.json"  # log file from analyze_droplets
CHUNK_SIZE_FRAMES = None    # chunk size in frames (set to None if using CHUNK_DURATION_S)
CHUNK_DURATION_S  = 5.0     # chunk duration in seconds (set to None if using CHUNK_SIZE_FRAMES)
START_UTC         = None    # UTC Unix timestamp of acquisition start (or None)
DENSITY_KG_M3     = 3520.0  # material density for size/mass calculations
```

Then run:
```bash
python longitudinal_analysis.py
```

## Chunking Strategy

**Frame-based chunks** (`CHUNK_SIZE_FRAMES`): Each chunk contains exactly that many frames. The last chunk may be smaller.

**Time-based chunks** (`CHUNK_DURATION_S`): The ideal chunk size is `duration × fps`. If this is not an integer, extra frames are distributed across chunks so that **no frames are left unanalyzed** and **each chunk differs by at most one frame** from the target duration.

## Outputs

### Per-chunk distribution plots
Saved to `plots/<basename>/chunk_XXXX.png`. Each plot is the full `analyze_distributions.analyze_independent()` figure for that chunk (counts, widths, speeds, projections, orientation rose).

### Longitudinal numpy file
`data/<basename>/longitudinal_analysis_<basename>_<YYYY-MM-DD_HH-MM-SS>.npy`

A structured NumPy array with one row per chunk. Load it with:
```python
import numpy as np
data = np.load("longitudinal_analysis_xenon_2026-03-11_09-16-43.npy", allow_pickle=False)
print(data.dtype.names)   # list all fields
```

#### Fields

| Field | Description |
|-------|-------------|
| `chunk_index` | Zero-based chunk index |
| `chunk_start_frame` / `chunk_end_frame` | Absolute frame numbers (inclusive) |
| `chunk_start_time_s` / `chunk_end_time_s` / `chunk_mid_time_s` | Chunk time bounds (seconds) |
| `n_droplets` | Total droplets detected in this chunk |
| `mean_count` / `std_count` | Mean and std of droplets per frame |
| `mean_density_mm3` / `std_density_mm3` | Volumetric density (droplets/mm³) |
| `mean_width_px` / `std_width_px` | Streak width (pixels) |
| `mean_width_um` / `std_width_um` | Streak width (µm) |
| `mean_speed_mps` / `std_speed_mps` | Droplet speed (m/s) |
| `mean_vy_mps` / `std_vy_mps` | Radial velocity component (m/s) |
| `mean_vz_mps` / `std_vz_mps` | Axial velocity component (m/s) |
| `wind_direction_deg` | Circular mean orientation angle (°, [0, 180)) |
| `anisotropy_ratio` | Max/min count in orientation histogram (1 = isotropic) |
| `start_utc` | UTC Unix timestamp of acquisition start (NaN if not provided) |

All statistics are **raw moments** (mean/std of the data) — no distribution fits are used.

#### Example plots
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("longitudinal_analysis_xenon_2026-03-11_09-16-43.npy", allow_pickle=False)

# Density vs time
plt.figure()
plt.plot(data['chunk_mid_time_s'], data['mean_density_mm3'], 'o-')
plt.xlabel('Time (s)')
plt.ylabel('Density (droplets/mm³)')
plt.show()

# Speed with error bars
plt.figure()
plt.errorbar(data['chunk_mid_time_s'], data['mean_speed_mps'] * 1e3,
             yerr=data['std_speed_mps'] * 1e3, fmt='o-')
plt.xlabel('Time (s)')
plt.ylabel('Speed (mm/s)')
plt.show()
```

### Longitudinal time-series plots
One 2-row figure per parameter group saved to `plots/<basename>/`:

| File | Content |
|------|---------|
| `<basename>_longitudinal_count.png` | Droplet count (+ density twin axis) vs time |
| `<basename>_longitudinal_width.png` | Streak width px (+ µm twin axis) vs time |
| `<basename>_longitudinal_speed.png` | Speed (mm/s) vs time |
| `<basename>_longitudinal_vy.png` | Radial velocity vs time |
| `<basename>_longitudinal_vz.png` | Axial velocity vs time |
| `<basename>_longitudinal_wind_direction.png` | Circular mean angle (+ anisotropy twin axis) vs time |

Top panel: the parameter value. Bottom panel: its standard deviation.

## Notes

- **Exposure time**: The velocity calculation uses `exposure_time_s` from the log if present; otherwise falls back to `1/fps`.
- **UTC timestamp**: Set `START_UTC` to enable correlation with external (e.g. environmental) data. Convert a `datetime` to a Unix timestamp with:
  ```python
  import datetime
  dt = datetime.datetime(2026, 3, 11, 9, 0, 0, tzinfo=datetime.timezone.utc)
  print(dt.timestamp())
  ```
- **Multiple runs**: Each run generates a uniquely timestamped numpy file in `data/<basename>/`, so previous results are never overwritten.
