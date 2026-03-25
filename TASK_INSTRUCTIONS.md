# Task Instructions (Temporary Reference File)

## Overview
Longitudinal analysis workflow for xenon microsphere droplet videos.

---

## Task 1: analyze_droplets.py — Log Files
- When `droplet_analysis` runs (i.e., when saving results), create a log file in `logs/` directory
- Log file name: `<video_basename>_<YYYY-MM-DD_HH-MM-SS>.json`
- Log file stores ALL parameters used in the analysis
- Add "Load Log" button in the GUI — clicking it opens a file dialog to load a log JSON and repopulate all GUI parameters

## Task 2: analyze_droplets.py — FPS/Exposure button
- Add a button (or textboxes) in the GUI for FPS override and exposure time (seconds)
- FPS determines the time axis (frame → seconds conversion)
- Exposure determines velocity from streak lengths: `velocity = streak_length / exposure_time`
- If exposure is left blank, fall back to `1/fps` (frame period) for velocity

## Task 3: Data Directory Structure
- Create `data/` directory at project root
- Each video gets a subfolder: `data/<video_basename>/`
- Files: `<basename>_droplet_data.npy`, `<basename>_test_results.json`, `<basename>_tracking_data.npy`
- Example: video `xenon.mp4` → `data/xenon/xenon_droplet_data.npy`, `data/xenon/xenon_test_results.json`
- Update all scripts to point to new paths

## Task 4: longitudinal_analysis.py
- Load a log file (from `logs/`) to get analysis parameters
- Define chunk sizes in FRAMES or TIME (seconds)
  - Smart rounding: if time-based chunk size is not integer multiple of fps, round some chunks up/down so no frames are left unanalyzed
  - It is fine for some chunks to have one frame more or less
- Partition the existing droplet data (by frame ranges) into chunks — no need to re-run detection
- Run `analyze_distributions.analyze_independent()` on each chunk
- Save a plot for each chunk in `plots/<video_basename>/` subfolder
- Extract 4 data points from each subplot (raw moments, NOT fits):
  - **Subplot 1 (counts/density)**: mean_count_per_frame, std_count_per_frame, mean_density_per_mm3, std_density_per_mm3
  - **Subplot 2 (width)**: mean_width_px, std_width_px, mean_width_um, std_width_um
  - **Subplot 3 (speed)**: v_mean_mps, v_std_mps, v_mean_mmps, v_std_mmps
  - **Subplot 4 (vy projection)**: mean_vy_mps, std_vy_mps, mean_vy_mmps, std_vy_mmps
  - **Subplot 5 (vz projection)**: mean_vz_mps, std_vz_mps, mean_vz_mmps, std_vz_mmps
  - **Polar wind plot**: circular_mean_angle_deg, anisotropy_ratio (direction and magnitude)
- Numpy file organization: structured so you can easily plot chunk_index vs density, etc.
- Store UTC start_time in numpy file (specified in `if __name__` section)
- Output numpy file: `data/<video_basename>/longitudinal_analysis_<basename>_<YYYY-MM-DD_HH-MM-SS>.npy`
- Save longitudinal plots (1 col, 2 rows): param on top, std dev on bottom
  - For count/density: twin y-axis showing number vs density
  - Save to `plots/<video_basename>/longitudinal_<param>.png`
- Create `longitudinal_analysis.md` README

## Task 5: environmental_correlations.py
- Point to a longitudinal data numpy file
- Read start_time and end_time from the file
- Use query_db.py to poll InfluxDB for:
  - RTD_1 → TEMP_CUBE
  - RTD_2 → TEMP_BASE
  - RTD_3 → TEMP_TOP
  - RTD_4 → TEMP_BOT
  - Setra 225 - Fill Line → PRESSURE_CUBE
  - Setra 225 - Xenon Bottle → PRESSURE_BOTTLE
- Create scatter plots: temperature vs longitudinal data, pressure vs longitudinal data
- Save plots in `plots/environmental_correlations/`

## Task 6: Update README.md
- Document new log file workflow
- Document FPS/exposure GUI controls
- Document new data directory structure
- Add new scripts to table: longitudinal_analysis.py, environmental_correlations.py
- Update quick start workflow

---

## Completion Checklist
- [ ] Log file creation in logs/ dir
- [ ] Load Log button in analyze_droplets GUI
- [ ] FPS/Exposure textboxes in GUI (fps_override, exposure_time_s)
- [ ] data/ directory + subfolders by video basename
- [ ] analyze_droplets.py uses data/<basename>/<basename>_*.npy paths
- [ ] track_droplets.py updated for new data paths
- [ ] analyze_distributions.py updated for new data paths
- [ ] longitudinal_analysis.py created
- [ ] longitudinal_analysis.md created
- [ ] environmental_correlations.py created
- [ ] README.md updated
