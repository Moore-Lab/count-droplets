# analyze_distributions.py

Performs statistical distribution analysis on droplet data produced by `analyze_droplets.py`. Fits physical models to droplet count, size, and speed distributions and reports calibrated results.

## Dependencies

```bash
pip install numpy matplotlib scipy
```

## Inputs

- **`droplet_data.npy`** (or `.npz`) - Structured array with fields: `frame`, `x`, `y`, `length`, `width`, `angle`.
- **`test_results.json`** (optional) - Metadata from `analyze_droplets.py` providing calibration values (`pixels_per_um`, `fps`, `exposure_time_s`, `beam_waist_px`, etc.).

## Usage

Edit the settings at the bottom of the script:

```python
data_file = "droplet_data.npy"
results_json = "test_results.json"
save_plot_step1 = None  # set to a path to save the figure instead of displaying
```

Run:

```bash
python analyze_distributions.py
```

### Programmatic Usage

```python
from analyze_distributions import load_droplet_data, analyze_independent, analyze_combined

per_droplet = load_droplet_data("droplet_data.npy", results_json="test_results.json")
step1 = analyze_independent(per_droplet, density_kg_m3=3520.0)
step2 = analyze_combined(step1, per_droplet)
```

## Thermal Speed Distributions and Dimensional Projection

### Starting point: the Maxwell--Boltzmann distribution

Consider a particle of mass `m` in thermal equilibrium at temperature `T`. Each Cartesian velocity component is independently drawn from a zero-mean Gaussian:

```
v_i ~ N(0, sigma),    sigma = sqrt(kT / m),    i in {x, y, z}
```

where `k` is the Boltzmann constant. The 3D speed `s = sqrt(v_x^2 + v_y^2 + v_z^2)` then follows the **Maxwell--Boltzmann speed distribution** (equivalently, a chi distribution with `k = 3` degrees of freedom):

```
f_3D(s; sigma) = sqrt(2/pi) * (s^2 / sigma^3) * exp(-s^2 / (2 * sigma^2)),   s >= 0
```

The `s^2` prefactor arises from the surface area element of a sphere in velocity space: `dOmega_3D = 4*pi*s^2`.

### Projection onto the image plane (3D to 2D)

A camera integrates along the line of sight (say the x-axis), collapsing 3D motion onto a 2D image plane spanned by (y, z). The line-of-sight component `v_x` is unobserved, and the projected 2D speed is:

```
u = sqrt(v_y^2 + v_z^2)
```

Since `v_y` and `v_z` remain independent Gaussians with the same `sigma`, `u` follows a **Rayleigh distribution** (chi distribution with `k = 2`):

```
f_2D(u; sigma) = (u / sigma^2) * exp(-u^2 / (2 * sigma^2)),   u >= 0
```

The `u` prefactor (replacing `s^2` in 3D) arises from the circumference of a circle in 2D velocity space: `dOmega_2D = 2*pi*u`. This is the distribution fit in Histogram 3.

Key properties:

| Quantity | Expression |
|----------|------------|
| Mode | `sigma` |
| Mean | `sigma * sqrt(pi/2)` |
| RMS | `sigma * sqrt(2)` |
| Variance | `sigma^2 * (2 - pi/2)` |

### Projection onto a single axis (2D to 1D)

Decomposing the 2D projected speed into components along the image axes gives the individual velocity components `v_y` and `v_z`, each distributed as `N(0, sigma)`. However, the streak angle from `minAreaRect` provides only the *orientation* of the streak (a line), not the *direction* of motion along it. The angle `theta` and `theta + pi` are indistinguishable, so the sign of `v_y = u * sin(theta)` and `v_z = u * cos(theta)` is arbitrary. The observable quantity is therefore the magnitude:

```
|v_i| = u * |cos(theta)|   or   u * |sin(theta)|
```

Since `v_i ~ N(0, sigma)`, the folded variable `|v_i|` follows a **half-normal distribution** (chi distribution with `k = 1`):

```
f_1D(x; sigma) = sqrt(2/pi) / sigma * exp(-x^2 / (2 * sigma^2)),   x >= 0
```

There is no speed-dependent prefactor (the "surface area" of a point in 1D velocity space is unity). This is the distribution fit in Histograms 4 and 5.

Key properties:

| Quantity | Expression |
|----------|------------|
| Mode | `0` |
| Mean | `sigma * sqrt(2/pi)` |
| RMS | `sigma` |
| Variance | `sigma^2 * (1 - 2/pi)` |

### Summary: the chi distribution hierarchy

All three speed distributions are special cases of the chi distribution with `k` degrees of freedom, parameterized by the same thermal velocity scale `sigma = sqrt(kT/m)`:

```
f_chi(s; k, sigma) = [2^(1-k/2) / (sigma * Gamma(k/2))] * (s/sigma)^(k-1) * exp(-s^2 / (2*sigma^2))
```

| k | Distribution | Observable | Prefactor origin | Fit in |
|---|-------------|-----------|-----------------|--------|
| 1 | Half-normal | \|v_y\| or \|v_z\| (single axis magnitude) | Unity (point) | Histograms 4, 5 |
| 2 | Rayleigh | u = sqrt(v_y^2 + v_z^2) (image-plane speed) | 2*pi*u (circle circumference) | Histogram 3 |
| 3 | Maxwell--Boltzmann | s = sqrt(v_x^2 + v_y^2 + v_z^2) (3D speed) | 4*pi*s^2 (sphere surface area) | Not directly observed |

The single shared parameter `sigma` connects all three distributions. For isotropic thermal motion, the fitted `sigma` values from Histograms 3, 4, and 5 should agree within statistical uncertainty. Disagreement between the Rayleigh `sigma` and either half-normal `sigma` may indicate anisotropic motion, bulk flow, or systematic bias in the streak angle measurement. Disagreement between the two half-normal `sigma` values (y vs z) specifically indicates anisotropy between the vertical and horizontal image axes.

### Relation to temperature

Given the fitted `sigma` and the droplet mass `m` derived from the streak width (Histogram 2), the effective temperature is:

```
T = m * sigma^2 / k
```

This relation applies identically regardless of which histogram's `sigma` is used, since all three distributions share the same `sigma` parameter. In practice, the Rayleigh fit (Histogram 3) uses all the data jointly and is typically the most precise estimator.

## Analysis Pipeline

### Step 1: Independent Analysis

Produces a 4x3 figure. Rows 1-2 contain 3 histograms (counts, width, speed magnitude) with their residual subplots. Rows 3-4 contain 2 projected speed histograms (|v_y|, |v_z|) with their residual subplots, plus a polar rose plot of streak orientations in the bottom-right cell (spanning both rows). All histograms use step style with sqrt(n) error bars. Residual subplots show normalized residuals `(O - E) / sqrt(E)` with +/-2 sigma reference lines. Reduced chi^2 is quoted in each legend.

#### Histogram 1: Droplet Number Density (Poisson)

**What is measured:** The number of droplets detected per frame. Each frame is one independent observation, so the distribution of counts across frames is histogrammed.

**Physical model:** Droplet arrivals in the laser beam cross-section are modeled as a Poisson process. The Poisson PMF `P(k; lambda) = lambda^k * exp(-lambda) / k!` is fit to the histogram via least-squares on the binned probability density. The single free parameter lambda represents the expected count per frame.

**Unit conversions:**
- The twin x-axis converts raw counts to volumetric density (droplets/mm^3) by dividing by the illuminated volume `V = pi * r^2 * z`, where `r` is the beam radius and `z` is the beam depth (ROI width). Volume is computed from either manually supplied `r_mm`/`z_mm` or inferred from ROI pixel dimensions and the `pixels_per_um` calibration.
- Areal density (droplets/mm^2) is also computed by dividing counts by the beam cross-sectional area `pi * beam_waist_px^2`, converted to mm^2.

**Reported values:** lambda (fit +/- error), mean count +/- std, volumetric density +/- std (/mm^3), chi^2_red.

#### Histogram 2: Streak Width Distribution (Poisson)

**What is measured:** The minor axis (width) of each droplet's minimum-area bounding rectangle, as determined by `analyze_droplets.py`. The width is the dimension perpendicular to the streak's motion direction. Since widths are quantized to pixel resolution, the distribution is discrete.

**Physical model:** The integer-rounded widths are fit with a Poisson distribution. This is an empirical choice reflecting the discrete, positive-valued nature of the data rather than a first-principles physical model.

**Unit conversions:** The twin x-axis converts pixels to microns via `width_um = width_px / pixels_per_um`.

**Derived quantities (width to mass):** The mean streak width is interpreted as the apparent diameter of a spherical droplet:

```
diameter_m  = mean_width_px / (pixels_per_um * 1e6)
radius_m    = diameter_m / 2
volume_m3   = (4/3) * pi * radius_m^3
mass_kg     = volume_m3 * density_kg_m3
```

The `density_kg_m3` parameter defaults to 3520 kg/m^3. This derived mass is passed to Step 2 and could be used for temperature fitting in future analysis.

**Reported values:** lambda (fit +/- error), mean width +/- std (px and um), derived radius, volume, mass, chi^2_red.

#### Histogram 3: Projected Speed Distribution (2D Rayleigh)

**What is measured:** The projected speed of each droplet, inferred from the streak length.

**From streak length to speed:** During a single camera exposure of duration `t_exp`, a moving droplet traces a streak whose length (major axis of the minimum-area rectangle) is proportional to its projected speed:

```
speed_m_s = (streak_length_px * px_to_m) / exposure_time_s
```

where `px_to_m = 1 / (pixels_per_um * 1e6)`. If `exposure_time_s` is not provided in the metadata, it defaults to `1 / fps`. The streak length is the *full* length (not length minus width), so this speed estimate includes the static droplet diameter. For streaks much longer than the droplet width this is a small correction.

**Physical model -- why a 2D Rayleigh distribution:** The camera images a 2D projection of 3D droplet motion. If each velocity component (vx, vy, vz) is independently drawn from a Gaussian with the same standard deviation sigma (as expected for thermal/Brownian motion), then the speed projected onto the image plane is:

```
u = sqrt(vx^2 + vy^2)       (if camera sees two components)
```

The magnitude of two independent Gaussian components follows a **Rayleigh distribution**:

```
f(u; sigma) = (u / sigma^2) * exp(-u^2 / (2 * sigma^2)),   u >= 0
```

This is a one-parameter family with mode at sigma and mean at `sigma * sqrt(pi/2)`.

**Fitting procedure:** The Rayleigh sigma is estimated via the maximum-likelihood estimator:

```
sigma_MLE = sqrt( mean(u^2) / 2 )
```

with standard error `sigma_err = sigma / sqrt(2n)`. The predicted PDF is then overlaid on the histogram and the residuals are computed.

**Reported values:** Rayleigh sigma +/- error (mm/s), mean speed from data +/- std (mm/s), Rayleigh predicted mean `sigma * sqrt(pi/2)` (mm/s), RMS speed, chi^2_red. The comparison between the data mean and Rayleigh predicted mean indicates how well the 2D Rayleigh model describes the observed speed distribution.

#### Histogram 4: Vertical Projected Speed |v_y| (Half-Normal)

**What is measured:** The magnitude of the speed component projected onto the vertical (y) image axis. Computed as `|v_y| = speed * |sin(angle)|`, where `angle` is the streak orientation from the minimum-area bounding rectangle and `speed` is the full projected speed from Histogram 3.

**Why absolute value:** The streak angle from `minAreaRect` gives the *orientation* of the streak (a line) but not the *direction* of motion along it. The angle and angle + 180 degrees are equivalent, so the sign of `sin(angle)` is arbitrary. Only the magnitude of the projection is physically meaningful without frame-to-frame tracking (which is handled separately in `track_droplets.py`).

**Physical model -- why a half-normal distribution:** If the underlying velocity component v_y is drawn from a zero-mean Gaussian N(0, sigma) (as expected for thermal motion with no bulk flow), then the absolute value |v_y| follows a **half-normal distribution**:

```
f(x; sigma) = sqrt(2/pi) / sigma * exp(-x^2 / (2 * sigma^2)),   x >= 0
```

This is a one-parameter family. The mean is `sigma * sqrt(2/pi)` and the variance is `sigma^2 * (1 - 2/pi)`.

**Fitting procedure:** The half-normal sigma is estimated via MLE:

```
sigma_MLE = sqrt( mean(x^2) )
```

with standard error `sigma_err = sigma / sqrt(2n)`.

**Reported values:** Half-normal sigma +/- error (mm/s), data mean (mm/s), fit predicted mean `sigma * sqrt(2/pi)` (mm/s), chi^2_red.

#### Histogram 5: Horizontal Projected Speed |v_z| (Half-Normal)

**What is measured:** The magnitude of the speed component projected onto the horizontal (z) image axis. Computed as `|v_z| = speed * |cos(angle)|`.

**Physical model:** Identical to Histogram 4: the underlying v_z component is modeled as N(0, sigma), and the observed |v_z| follows a half-normal distribution.

**Consistency check:** For isotropic thermal motion, the half-normal sigma values from Histograms 4 and 5 should be equal (within statistical uncertainty) and should both equal the Rayleigh sigma from Histogram 3. Significant disagreement may indicate anisotropic motion, bulk flow, or systematic bias in the streak angle measurement.

**Reported values:** Half-normal sigma +/- error (mm/s), data mean (mm/s), fit predicted mean (mm/s), chi^2_red.

#### Plot 6: Orientation Rose (Polar)

**What is measured:** The distribution of streak orientations across angular bins, displayed as a polar rose diagram. This plot occupies the bottom-right cell, spanning both the histogram and residual rows.

**Construction:** Streak angles from `minAreaRect` are folded into [0, 180) degrees (since orientation and orientation + 180 degrees are equivalent). The folded range is divided into angular bins (default 12 bins of 15 degrees). For visual symmetry, the histogram is mirrored to the full [0, 360) circle. Two quantities are plotted:

- **Count density** (blue bars): The number of streaks per angular bin, normalized to probability density. For isotropic motion, this should be uniform — shown as a dashed black reference circle at the level `1/pi`.
- **Speed-weighted overlay** (orange bars): The mean speed of droplets within each angular bin, scaled to fit the same radial axis. This reveals whether fast-moving droplets prefer certain orientations, which would indicate directional bulk flow.

**Anisotropy ratio:** The ratio `max(count_bins) / min(count_bins)` is reported as a scalar summary. A value of 1.0 indicates perfect isotropy; values significantly above 1 indicate preferred orientations. This is a coarse diagnostic — for detailed directional analysis with signed velocity vectors, see the ideas section in `track_droplets.md`.

**Interpreting the rose plot:**
- A uniform circle → isotropic thermal motion, no preferred direction
- Lobes at 0°/180° (horizontal) → preferred horizontal flow (e.g., convective currents)
- Lobes at 90°/270° (vertical) → preferred vertical motion (e.g., gravity-driven settling, buoyant plumes)
- Speed-weighted lobes that differ from count lobes → fast droplets have a different directional preference than the overall population, suggesting superimposed bulk flow
- The axis labels indicate 0° = z (horizontal) and 90° = y (vertical) to match Histograms 4 and 5

**Limitations:** Since `analyze_distributions.py` only has streak orientation (not direction of motion), the rose plot cannot distinguish between flow going left vs right or up vs down. It detects *axial* preference (e.g., "horizontal motion is more common") but not *directional* preference (e.g., "leftward flow"). For directional analysis, use `track_droplets.py` which preserves signed velocity vectors from frame-to-frame linking.

**Reported values:** Anisotropy ratio (max/min count), number of angular bins.

### Step 2: Combined Analysis

Currently a placeholder that passes through the width-derived mass and Rayleigh sigma from Step 1. Reserved for future joint analysis (e.g., Boltzmann temperature fitting using both mass and speed).

## Key Parameters

| Parameter | Default | Source | Description |
|-----------|---------|--------|-------------|
| `pixels_per_um` | 0.1878 | metadata | Spatial calibration (pixels per micron) |
| `fps` | 30.0 | metadata | Frame rate |
| `exposure_time_s` | 1/fps | metadata | Camera exposure time per frame |
| `beam_waist_px` | 175.0 | metadata | Beam waist radius for area calculation |
| `density_kg_m3` | 3520.0 | argument | Material density for mass calculation |
| `r_mm` / `z_mm` | inferred | argument | Manual override for illuminated volume geometry |

Volume geometry can be set manually with `r_mm` and `z_mm` arguments to `analyze_independent()`, or it is inferred from the ROI pixel dimensions: `r = 0.5 * roi_height * mm_per_px`, `z = roi_width * mm_per_px`.

## Console Output

Prints a summary including:
- Frames analyzed, exposure time, beam area, volume factor
- Mean droplets per frame with Poisson chi^2
- Areal density (/mm^2) and volumetric density (/mm^3)
- Average width in pixels and microns
- Average speed (data) and Rayleigh fit parameters
- |v_y| and |v_z| half-normal sigma values
- Orientation anisotropy ratio
- Resolved mass and sigma from the combined step

## Output

`analyze_independent()` returns a dictionary with keys:

- **`metadata_used`** - calibration values applied (`pixels_per_um`, `fps`, `exposure_time_s`, `beam_waist_px`, `beam_area_mm2`, `density_kg_m3`, `volume_mm3`, etc.)
- **`counts`** - `lambda_fit`, `lambda_fit_err`, `mean_per_frame`, `std_per_frame`, `mean_density_per_mm2`, `mean_density_per_mm3`, `chi2_reduced`, `hist_bins`
- **`width`** - `mean_width_px`, `std_width_px`, `mean_width_um`, `std_width_um`, `lambda_fit`, `radius_m`, `volume_m3`, `mass_kg`, `chi2_reduced`, `hist_bins`
- **`speed`** - `v_mean_mps`, `v_std_mps`, `v_rms_mps`, `sigma_fit_mps`, `sigma_fit_err_mps`, `rayleigh_mean_predicted_mps`, `chi2_reduced`, `speed_bins_mps`
- **`speed_y_proj`** - `sigma_fit_mps`, `sigma_fit_err_mps`, `chi2_reduced`
- **`speed_z_proj`** - `sigma_fit_mps`, `sigma_fit_err_mps`, `chi2_reduced`
- **`orientation`** - `anisotropy_ratio`, `n_angle_bins`
