# analyze_tracks.py

Statistical analysis of tracked droplet trajectories to characterize flow anisotropy, bulk flow velocity, and directional speed distributions. Operates on the signed velocity vectors available from frame-to-frame tracking, which `analyze_distributions.py` cannot access since it only has streak orientation.

## Dependencies

```bash
pip install numpy matplotlib
```

## Inputs

- **`track_data.npz`** — Saved by `track_droplets.save_track_data()`. Contains a structured array of per-observation records with pre-computed velocity fields, plus JSON metadata.

### track_data.npz Format

**`track_observations`** structured array (one row per track-frame observation):

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int32 | Track identifier |
| `frame` | int32 | Frame number |
| `start_x`, `start_y` | float32 | Streak start point (px) |
| `end_x`, `end_y` | float32 | Streak end point (px) |
| `vx`, `vy` | float32 | Velocity components (px/s), from linking displacement `prev_end → curr_start` divided by inter-frame time. NaN for first observation per track. |
| `speed` | float32 | √(vx² + vy²) (px/s) |
| `direction` | float32 | atan2(vy, vx) mapped to [0, 360°) |

**`metadata`** JSON string with: `fps`, `pixels_per_um`, `max_distance`, `min_track_length`, `n_tracks`, `n_observations`.

## Usage

### Standalone

```bash
python analyze_tracks.py
```

Loads `track_data.npz` from the current directory.

### Called from track_droplets.py

`track_droplets.main()` automatically saves `track_data.npz` and calls `analyze_tracks` at the end of its pipeline:

```python
save_track_data(endpoints_by_track, "track_data.npz", fps=fps, pixels_per_um=pixels_per_um, ...)
import analyze_tracks
track_data = analyze_tracks.load_track_data("track_data.npz")
analyze_tracks.analyze_tracks(track_data)
```

### Programmatic

```python
from analyze_tracks import load_track_data, analyze_tracks

track_data = load_track_data("track_data.npz")
results = analyze_tracks(track_data, n_angle_bins=36, n_time_windows=10)
```

## Analysis Panels

The figure is a 3×2 grid with 5 analysis panels and a summary text panel.

### Panel 1: Speed–Direction Heatmap

A 2D histogram with direction angle (0–360°) on the x-axis and speed on the y-axis. Unlike the orientation rose in `analyze_distributions.py`, tracked droplets have a defined direction of motion (not just axis), so the full [0, 360) range is available.

**Interpretation:**
- Convective flow → concentrated hot spot at a preferred angle and elevated speed
- Turbulent flow → diffuse band across multiple angles
- Gravity settling → narrow peak near 270° (downward) at low speed
- Isotropic thermal → uniform ring at all angles

### Panel 2: Velocity Scatter with Anisotropy Ellipse

Each tracked displacement plotted as a point in (vx, vy) space. A 2D Gaussian is fit via the sample covariance matrix, and 1σ / 2σ confidence ellipses are drawn.

**Key quantities annotated:**
- **Centroid offset** (red cross) = bulk flow velocity vector
- **Eccentricity** — 0 = circular (isotropic), 1 = maximally elongated
- **Tilt angle** — orientation of the major axis
- **σ_major, σ_minor** — principal dispersions

**Interpretation:**
- Circular cloud at origin → isotropic thermal motion, no bulk flow
- Elliptical cloud at origin → anisotropic thermal motion (geometry/confinement)
- Cloud offset from origin → bulk flow superimposed on thermal; the offset vector gives the mean flow velocity
- Elliptical cloud offset from origin → anisotropic motion plus bulk flow

### Panel 3: Angular Speed Profile (Polar)

Polar plot of mean speed vs direction in angular bins. A dashed reference circle shows the global mean speed.

**Interpretation:**
- Circle → isotropic speed in all directions
- Peanut shape → faster motion along one axis but no preferred direction
- Cardioid/offset circle → bulk flow; the bulge points in the flow direction

### Panel 4: Spatial Flow Field (Quiver)

The frame is divided into a spatial grid. Within each cell, the mean velocity vector is computed and displayed as an arrow. Background shading indicates observation density. Only cells with ≥ `min_obs_per_bin` observations are plotted.

**Interpretation:**
- Reveals convective cell structure (circulation patterns, stagnation points)
- Shows whether flow is localized or global
- Coherent vs incoherent neighboring arrows → laminar vs turbulent

### Panel 5: Temporal Evolution of Anisotropy

The frame range is divided into `n_time_windows` equal windows. Within each, the anisotropy ellipse is refit and two time series are plotted:
- **Eccentricity** (left axis, red) — shape of the velocity distribution
- **Bulk flow speed** (right axis, blue) — magnitude of the mean velocity offset

**Interpretation:**
- Monotonic trends → evolving conditions (heating, cooling)
- Oscillations → periodic convective cells or acoustic modes
- Step changes → onset/cessation of external perturbation
- Convergence toward low eccentricity + low bulk flow → successful minimization of convective disturbances

### Panel 6: Summary Statistics

Text panel displaying key numerical results for quick reference.

## Output

`analyze_tracks()` returns a dict:

```python
{
    "metadata_used": { ... },
    "velocity_stats": {
        "n_tracks": int,
        "n_velocity_measurements": int,
        "mean_speed_mps": float,
        "rms_speed_mps": float,
        "std_speed_mps": float,
        "mean_speed_mmps": float,
    },
    "anisotropy_ellipse": {
        "centroid_vx": float,
        "centroid_vy": float,
        "bulk_flow_speed": float,
        "bulk_flow_direction_deg": float,
        "sigma_major": float,
        "sigma_minor": float,
        "eccentricity": float,
        "tilt_angle_deg": float,
        "sigma_ratio": float,
    },
    "angular_profile": {
        "bin_centers_deg": list,
        "mean_speed_mmps_per_bin": list,
    },
    "temporal_evolution": {        # present if n_time_windows > 0
        "window_centers_frame": list,
        "eccentricity_vs_time": list,
        "bulk_flow_vs_time": list,
        "sigma_ratio_vs_time": list,
    },
}
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_angle_bins` | 24 | Number of angular bins (heatmap + polar profile) |
| `n_time_windows` | 5 | Number of temporal windows for evolution panel |
| `spatial_bins` | 8 | Grid divisions per axis for quiver plot |
| `min_obs_per_bin` | 3 | Minimum observations required to populate a spatial/angular bin |

## Velocity Convention

The velocity for each track step is computed from the **linking displacement**:

```
displacement = curr_start - prev_end
velocity = displacement / (n_frames_gap / fps)
```

where `curr_start` is the start point of the current detection and `prev_end` is the end point of the previous detection in the same track. This represents the inter-frame motion of the droplet. The first observation in each track has no predecessor and receives NaN velocity fields.

The direction angle uses the standard mathematical convention: `atan2(vy, vx)` mapped to [0, 360°), where 0° = rightward (+x), 90° = downward (+y in image coordinates).
