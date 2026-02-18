# track_droplets.py

Links droplet detections across consecutive frames to build multi-frame tracks, then visualizes the resulting trajectories as vectors, streamlines, and summed-image overlays.

## Dependencies

```bash
pip install numpy matplotlib opencv-python
```

## Inputs

- **`droplet_data.npy`** - Structured NumPy array produced by `analyze_droplets.py`. Required fields: `frame`, `x`, `y`, `length`, `width`, `angle`. Endpoint fields (`length_start_x/y`, `length_end_x/y`) are computed automatically if missing.
- **Video file** (optional) - Required only for summed-image track visualization.

## Usage

Edit the settings in the `if __name__ == '__main__'` block:

```python
input_file = "droplet_data.npy"
video_path = "your_video.mp4"    # needed for summed track plots
canvas = (1920, 1080)            # frame resolution
max_distance = 24                # linking radius in pixels
min_track_length = 1             # minimum frames per track
plot_summed_track_id = 5         # track ID for summed image (None to skip)
summed_padding = 50              # pixels around track bounding box
```

Run:

```bash
python track_droplets.py
```

### Programmatic Usage

```python
from track_droplets import load_tracking_data, link_droplets_by_endpoints, main

# Quick: run the full pipeline
endpoints = main(input_file="droplet_data.npy", max_distance=24)

# Manual: load, link, then visualize
data = load_tracking_data("droplet_data.npy")
tracks = link_droplets_by_endpoints(data, max_distance=24)
```

## Linking Algorithm

Droplets are linked across consecutive frames using endpoint proximity with greedy one-to-one matching:

1. For each pair of consecutive frames, compute the distance from each previous detection's **end point** to both endpoints of each current detection.
2. For the **first link of a track**, the algorithm also considers the previous detection's **start point** (to correct initial orientation ambiguity).
3. Accept candidates within `max_distance` pixels.
4. Sort by distance and greedily assign one-to-one matches (no merges/splits).
5. Swap current detection orientation if needed so that links are always `prev_end -> curr_start`.
6. Unmatched detections start new tracks.

### Automatic Threshold Selection

Set `max_distance = None` to automatically scan thresholds from 0-74 pixels. The script calls `set_distance_threshold.py` to fit the tracked-droplets-vs-distance curve and suggests an optimal threshold.

## Visualizations

### Endpoint Vectors

Draws per-frame streak vectors (start -> end) for each track, color-coded by track ID. Controlled by `plot_first_n` (default 10 tracks).

### Streamlines

Concatenates streak start/end points into continuous polylines per track, with an arrowhead on the last segment showing direction of motion.

### Summed Track Image

For a specific track ID, sums the raw video frames where the droplet appears, cropped around the track bounding box, and overlays the streamline or individual vectors in cyan. Requires `video_path`.

```python
# Plot summed image for track 5
main(input_file="droplet_data.npy", video_path="video.mp4",
     plot_summed_track_id=5, summed_padding=50)
```

### Batch Summed Images

```python
from track_droplets import plot_all_tracks_summed

plot_all_tracks_summed(
    "video.mp4", endpoints_dict,
    min_track_length=2,   # only tracks with 2+ frames
    max_tracks=10,        # plot the 10 longest tracks
    save_dir="track_imgs" # save PNGs to directory
)
```

## Output

The `main()` function returns `endpoints_by_track`, a dictionary:

```python
{
    track_id: [
        {"frame": int, "start": (sx, sy), "end": (ex, ey)},
        ...
    ],
    ...
}
```

Each track is sorted by frame number. Track IDs are integers starting from 0.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_distance` | 24 px | Maximum linking distance between consecutive-frame endpoints |
| `min_track_length` | 1 | Minimum number of frames for a track to be kept |
| `canvas` | (1920, 1080) | Frame dimensions for plot axes |
| `plot_first_n` | 10 | Number of tracks to display in vector/streamline plots |
| `summed_padding` | 50 px | Pixel padding around track bounding box in summed images |

## Future Analysis Ideas: Anisotropy and Flow Characterization

The following analyses leverage the signed velocity vectors available from frame-to-frame tracking (which `analyze_distributions.py` cannot access since it only has streak orientation). These are candidates for future implementation in this script or a dedicated analysis module.

### Speed--angle heatmap (2D histogram)

A 2D histogram with signed direction angle on the x-axis and speed on the y-axis. Unlike the orientation rose in `analyze_distributions.py`, tracked droplets have a defined direction of motion (not just axis), so the full [0, 360) range is available. This reveals the joint distribution of speed and direction:

- Convective flow appears as a concentrated hot spot at a preferred angle and elevated speed
- Turbulent flow broadens the spot into a diffuse band
- Laminar-to-turbulent transition shows up as a tight lobe at moderate speeds broadening at higher speeds
- Gravity-driven settling produces a narrow peak near 270° (downward) at low speed, distinct from high-speed thermal/convective motion

### Velocity scatter plot with anisotropy ellipse

Plot each tracked displacement as a point in (v_z, v_y) space (signed components). Fit a 2D Gaussian (or confidence ellipse) to the cloud:

- **Circular cloud centered at origin** → isotropic thermal motion, no bulk flow
- **Elliptical cloud centered at origin** → anisotropic thermal motion (different sigma_y vs sigma_z), possibly from chamber geometry
- **Circular cloud offset from origin** → isotropic thermal motion with superimposed uniform bulk flow; the offset vector gives the flow velocity
- **Elliptical cloud offset from origin** → anisotropic motion plus bulk flow

Quantitative outputs: eccentricity, tilt angle, centroid offset (bulk flow velocity vector), principal sigma values. The centroid offset is particularly useful — it directly measures the mean flow velocity that the orientation-only analysis in `analyze_distributions.py` cannot detect.

### Angular sigma profile (directional speed map)

Divide the full [0, 360) direction range into angular bins (e.g., 24 bins of 15°). Within each bin, compute the mean and standard deviation of the speed. Plot as a polar curve where radius = mean speed at that direction:

- A circle → isotropic speed scale in all directions
- A peanut shape → faster motion along one axis (e.g., horizontal convection) but no preferred direction
- A cardioid or offset circle → bulk flow superimposed on thermal motion; the bulge points in the flow direction

This is the directional analog of the orientation rose, but with signed direction and speed magnitude rather than just count.

### Temporal evolution of anisotropy

Split tracked data into time windows (e.g., groups of 50-100 frames) and compute the anisotropy metrics (ellipse eccentricity, sigma ratio, mean flow vector) within each window. Plot as time series:

- Monotonic trends → evolving conditions (heating, cooling, pressure changes)
- Oscillations → periodic convective cells or acoustic modes
- Step changes → onset/cessation of external perturbation
- Convergence toward isotropy → successful minimization of convective disturbances

This is essential for the goal of minimizing non-isotropic behavior: it provides a feedback signal showing whether experimental adjustments (temperature control, vibration isolation, chamber geometry changes) are reducing the flow.

### Spatial flow field

For tracks with sufficient spatial coverage, bin by (x, y) position and compute the mean velocity vector in each spatial bin. Display as a quiver plot overlaid on the camera frame:

- Reveals convective cell structure (circulation patterns, stagnation points)
- Shows whether flow is localized (e.g., near a heat source) or global
- Identifies regions of laminar vs turbulent flow by the coherence of neighboring vectors
