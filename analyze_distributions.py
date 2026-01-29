
"""
Two-Step Distribution Analysis for Droplet Data (Simplified & Reordered)

Step 1 (Independent Analysis)
-----------------------------
- Histogram 1: Droplet counts per frame (Poisson overlay); twin axis shows density (/mm^3).
- Histogram 2: Streak WIDTH (pixels); twin axis to um; mean width -> radius -> volume -> mass
- Histogram 3: Streak LENGTH -> speed (m/s); Rayleigh distribution fit

All histograms use step style with sqrt(n) error bars and residual subplots.
Reduced chi2 is quoted in legends.

Step 2 (Combined Analysis - Scaffold)
------------------------------------
- Accepts Step 1 results and (optionally) raw data/metadata.
- For now, returns pass-through + placeholders.

Inputs
------
- droplet_data.[npz|npy]  (npz with named arrays or structured npy with fields)
  required fields: frame, x, y, length, width, angle
- test_results.json (optional): metadata
  keys used: pixels_per_um, fps, exposure_time_s, beam_waist_px (or beam_radius_px), frames_analyzed
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional, Tuple
from scipy.stats import poisson
from scipy.optimize import curve_fit


def fit_poisson(data: np.ndarray) -> Tuple[float, float]:
    """
    Fit a Poisson distribution to discrete count data using least-squares
    on the histogram.
    """
    data = np.asarray(data, dtype=int)
    if data.size == 0:
        return np.nan, np.nan

    bins = np.arange(int(np.min(data)), int(np.max(data)) + 2, 1)
    hist, edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    def poisson_pmf(k, lam):
        return poisson.pmf(np.round(k).astype(int), lam)

    lam_init = float(np.mean(data))

    try:
        popt, pcov = curve_fit(poisson_pmf, bin_centers, hist,
                               p0=[lam_init], bounds=(0.01, np.inf), maxfev=5000)
        lam_fit = popt[0]
        lam_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0.0
    except Exception:
        lam_fit = lam_init
        lam_err = np.nan

    return lam_fit, lam_err


# =============================================================================
# Loading helpers
# =============================================================================

def _load_numpy_container(filepath: str) -> Dict[str, Any]:
    loaded = np.load(filepath, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    arr = loaded
    if hasattr(arr, 'dtype') and arr.dtype.names:
        return {name: arr[name] for name in arr.dtype.names}
    raise ValueError("Provide a .npz with named arrays or a structured .npy with fields.")


def load_droplet_data(filepath: str, results_json: Optional[str] = None) -> Dict[str, Any]:
    D = _load_numpy_container(filepath)
    frames  = D['frame']
    x       = D['x']
    y       = D['y']
    lengths = D['length']
    widths  = D['width']
    angles  = D['angle']

    unique_frames, counts_per_frame = np.unique(frames, return_counts=True)
    metadata: Dict[str, Any] = {}
    if results_json is not None:
        with open(results_json, "r") as f:
            summary = json.load(f)
        metadata = summary.get("analysis_parameters", {})
        metadata.update(summary.get("results", {}))
        metadata["frames_analyzed"] = int(len(unique_frames))

    return {
        "counts": counts_per_frame,
        "lengths": lengths,
        "widths": widths,
        "angles": angles,
        "x": x,
        "y": y,
        "metadata": metadata
    }


# =============================================================================
# Small helpers
# =============================================================================

def resolve_exposure_time(metadata: dict) -> float:
    fps = float(metadata.get('fps', 30.0))
    if 'exposure_time_s' in metadata and metadata['exposure_time_s'] is not None:
        exposure_time_s = float(metadata['exposure_time_s'])
        print(f"[exposure] Using exposure_time_s from metadata: {exposure_time_s:.6f} s")
        return exposure_time_s
    exposure_time_s = 1.0 / fps
    print(f"[exposure] No 'exposure_time_s' in metadata; falling back to 1/fps = {exposure_time_s:.6f} s (fps={fps:.3f})")
    return exposure_time_s


def compute_chi2_residuals(observed: np.ndarray, expected: np.ndarray, n_params: int = 1) -> Tuple[np.ndarray, float, int]:
    """Compute normalized residuals and reduced chi2."""
    mask = expected > 0
    residuals = np.zeros_like(observed, dtype=float)
    residuals[mask] = (observed[mask] - expected[mask]) / np.sqrt(expected[mask])

    chi2 = np.sum(residuals[mask]**2)
    dof = int(np.sum(mask)) - n_params
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return residuals, chi2_red, dof


def rayleigh_2d_pdf(u: np.ndarray, sigma: float) -> np.ndarray:
    """2D Rayleigh PDF: f(u) = (u / sigma^2) * exp(-u^2 / (2 sigma^2))"""
    u = np.asarray(u, float)
    if sigma <= 0:
        return np.zeros_like(u)
    sigma2 = sigma**2
    pdf = (u / sigma2) * np.exp(-u**2 / (2 * sigma2))
    pdf = np.where(u >= 0, pdf, 0.0)
    return pdf


def fit_rayleigh(data: np.ndarray) -> Tuple[float, float]:
    """Fit a 2D Rayleigh distribution to speed data using MLE."""
    data = np.asarray(data, float)
    data = data[np.isfinite(data) & (data >= 0)]
    if data.size == 0:
        return np.nan, np.nan

    # MLE for Rayleigh: sigma = sqrt(sum(u^2) / (2n))
    sigma_fit = np.sqrt(np.mean(data**2) / 2.0)
    sigma_err = sigma_fit / np.sqrt(2.0 * data.size)

    return sigma_fit, sigma_err


# =============================================================================
# STEP 1: Independent analysis (plots + per-channel results)
# =============================================================================

def analyze_independent(
    per_droplet: Dict[str, Any],
    *,
    density_kg_m3: float = 3520.0,
    save_path: Optional[str] = None,
    r_mm: Optional[float] = None,
    z_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze counts, width, and speed independently.

    Features:
    - Step-style histograms with sqrt(n) error bars
    - Residual subplots below each histogram
    - Reduced chi2 quoted in legends
    - Speed plot shows data mean vs Rayleigh predicted mean
    """
    counts  = np.asarray(per_droplet['counts'])
    lengths = np.asarray(per_droplet['lengths'])
    widths  = np.asarray(per_droplet['widths'])
    metadata = per_droplet.get('metadata', {})

    # Calibration
    pixels_per_um = float(metadata.get('pixels_per_um', 0.1878))
    fps = float(metadata.get('fps', 30.0))
    exposure_time_s = resolve_exposure_time(metadata)

    # Volume factor (pi r^2 z) in mm^3
    mm_per_px = 1.0 / (pixels_per_um * 1000.0)

    if (r_mm is not None) and (z_mm is not None):
        r_used_mm = float(r_mm)
        z_used_mm = float(z_mm)
        volume_mm3 = float(np.pi * (r_used_mm ** 2) * z_used_mm)
        volume_source = "manual_rz_mm"
    else:
        roi_h_px = int(metadata.get('roi_h_px', metadata.get('frame_height_px', 0)))
        roi_w_px = int(metadata.get('roi_w_px', metadata.get('frame_width_px', 0)))

        if (roi_h_px <= 0) and (per_droplet.get('y') is not None) and (per_droplet['y'].size > 0):
            roi_h_px = int(np.max(per_droplet['y']) - np.min(per_droplet['y']) + 1)
        if (roi_w_px <= 0) and (per_droplet.get('x') is not None) and (per_droplet['x'].size > 0):
            roi_w_px = int(np.max(per_droplet['x']) - np.min(per_droplet['x']) + 1)

        r_used_mm = 0.5 * roi_h_px * mm_per_px
        z_used_mm = 1.0 * roi_w_px * mm_per_px
        volume_mm3 = float(np.pi * (r_used_mm ** 2) * z_used_mm)
        volume_source = "roi_pixels_to_mm"

    print(f"[volume] source={volume_source}, r={r_used_mm:.4f} mm, z={z_used_mm:.4f} mm, V={volume_mm3:.6f} mm^3")

    beam_waist_px = float(metadata.get('beam_waist_px', metadata.get('beam_radius_px', 175.0)))
    frames_analyzed = int(metadata.get('frames_analyzed', counts.size if counts.size > 0 else 0))

    pixels_per_m = pixels_per_um * 1e6
    px_to_m = 1.0 / pixels_per_m

    beam_cross_section_mm2 = (np.pi * beam_waist_px**2) / (pixels_per_um**2) / 1e6

    # --- Figure: 2 rows x 3 cols (top=histograms, bottom=residuals) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2, ax3 = axes[0]
    ax1_res, ax2_res, ax3_res = axes[1]

    # ============================================================
    # 1) Counts per frame (Poisson) + residuals
    # ============================================================
    mean_count = float(np.mean(counts)) if counts.size > 0 else np.nan
    std_count  = float(np.std(counts))  if counts.size > 0 else np.nan
    mean_count_per_mm2 = mean_count / beam_cross_section_mm2 if np.isfinite(mean_count) else np.nan
    std_count_per_mm2  = std_count  / beam_cross_section_mm2 if np.isfinite(std_count)  else np.nan
    mean_density_per_mm3 = mean_count / volume_mm3 if (np.isfinite(mean_count) and volume_mm3 > 0) else np.nan
    std_density_per_mm3  = std_count  / volume_mm3 if (np.isfinite(std_count)  and volume_mm3 > 0) else np.nan

    lam, lam_err, chi2_red_counts = np.nan, np.nan, np.nan

    if counts.size > 0:
        count_bins = np.arange(int(np.min(counts)), int(np.max(counts)) + 2, 1)
        bin_centers = (count_bins[:-1] + count_bins[1:]) / 2.0

        hist_counts, _ = np.histogram(counts, bins=count_bins)
        n_total = counts.size

        lam, lam_err = fit_poisson(counts)
        expected_counts = n_total * poisson.pmf(bin_centers.astype(int), lam)

        residuals, chi2_red_counts, _ = compute_chi2_residuals(hist_counts, expected_counts, n_params=1)

        errors = np.sqrt(hist_counts)
        errors[errors == 0] = 1

        bin_width = count_bins[1] - count_bins[0]
        hist_density = hist_counts / (n_total * bin_width)
        errors_density = errors / (n_total * bin_width)

        # Step histogram
        ax1.stairs(hist_density, count_bins, color='steelblue', linewidth=2, label='Data')
        ax1.errorbar(bin_centers, hist_density, yerr=errors_density,
                     fmt='none', ecolor='steelblue', capsize=3, capthick=1.5, elinewidth=1.5)

        # Poisson fit (discrete points)
        xk = np.arange(int(np.min(counts)), int(np.max(counts)) + 1)
        yk = poisson.pmf(xk, lam)
        ax1.plot(xk, yk, 'ro-', lw=2, markersize=6,
                 label=f'Poisson: $\\lambda$ = {lam:.2f} +/- {lam_err:.2f}\n$\\chi^2_{{red}}$ = {chi2_red_counts:.2f}')

        ax1.axvline(mean_count, color='darkblue', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_count:.2f} +/- {std_count:.2f} counts\n'
                          f'      {mean_density_per_mm3:.2f} +/- {std_density_per_mm3:.2f} /mm^3')

        ax1.legend(fontsize=9)
        ax1.set_xlim(np.min(counts) - 0.5, np.max(counts) + 0.5)
        ax1.set_ylim(0, np.max(hist_density) * 1.3)

        # Residuals subplot
        ax1_res.bar(bin_centers, residuals, width=0.8*bin_width, color='steelblue', alpha=0.7, edgecolor='black')
        ax1_res.axhline(0, color='black', linestyle='-', linewidth=1)
        ax1_res.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1_res.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1_res.set_xlim(ax1.get_xlim())
        ax1_res.set_ylabel('Residuals\n$(O-E)/\\sqrt{E}$', fontsize=10)
        ax1_res.set_xlabel('Droplet Count per Frame', fontsize=12)
        ax1_res.grid(True, alpha=0.3)
    else:
        count_bins = [0, 1]

    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Droplet Number Density', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    ax1_t = ax1.twiny()
    xlim = ax1.get_xlim()
    ax1_t.set_xlim(xlim[0] / volume_mm3, xlim[1] / volume_mm3)
    ax1_t.set_xlabel('Density (droplets / mm^3)', fontsize=11, color='steelblue')
    ax1_t.tick_params(axis='x', labelcolor='steelblue')

    # ============================================================
    # 2) WIDTH histogram + residuals
    # ============================================================
    lambda_width, lambda_width_err, chi2_red_width = np.nan, np.nan, np.nan

    if widths.size > 0:
        wmin, wmax = int(np.floor(np.min(widths))), int(np.ceil(np.max(widths)))
        width_bins = np.arange(wmin, wmax + 2, 1)
        bin_centers_w = (width_bins[:-1] + width_bins[1:]) / 2.0

        hist_width, _ = np.histogram(widths, bins=width_bins)
        n_total_w = widths.size

        widths_int = np.round(widths).astype(int)
        lambda_width, lambda_width_err = fit_poisson(widths_int)

        expected_width = n_total_w * poisson.pmf(bin_centers_w.astype(int), lambda_width)
        residuals_w, chi2_red_width, _ = compute_chi2_residuals(hist_width, expected_width, n_params=1)

        errors_w = np.sqrt(hist_width)
        errors_w[errors_w == 0] = 1

        bin_width_w = width_bins[1] - width_bins[0]
        hist_density_w = hist_width / (n_total_w * bin_width_w)
        errors_density_w = errors_w / (n_total_w * bin_width_w)

        # Step histogram
        ax2.stairs(hist_density_w, width_bins, color='orange', linewidth=2, label='Data')
        ax2.errorbar(bin_centers_w, hist_density_w, yerr=errors_density_w,
                     fmt='none', ecolor='orange', capsize=3, capthick=1.5, elinewidth=1.5)

        # Poisson fit (discrete points)
        xw = np.arange(wmin, wmax + 1)
        yw = poisson.pmf(xw, lambda_width)
        ax2.plot(xw, yw, 'ro-', lw=2, markersize=6,
                 label=f'Poisson: $\\lambda$ = {lambda_width:.2f} +/- {lambda_width_err:.2f}\n$\\chi^2_{{red}}$ = {chi2_red_width:.2f}')

        mean_w = float(np.mean(widths))
        std_w  = float(np.std(widths))
        mean_w_um = mean_w / pixels_per_um
        std_w_um  = std_w  / pixels_per_um

        ax2.axvline(mean_w, color='darkorange', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_w:.2f} +/- {std_w:.2f} px\n'
                          f'      {mean_w_um:.2f} +/- {std_w_um:.2f} um')

        ax2.legend(fontsize=9)
        ax2.set_xlim(wmin - 0.5, wmax + 0.5)
        ax2.set_ylim(0, np.max(hist_density_w) * 1.3)

        # Residuals subplot
        ax2_res.bar(bin_centers_w, residuals_w, width=0.8*bin_width_w, color='orange', alpha=0.7, edgecolor='black')
        ax2_res.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2_res.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2_res.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2_res.set_xlim(ax2.get_xlim())
        ax2_res.set_ylabel('Residuals\n$(O-E)/\\sqrt{E}$', fontsize=10)
        ax2_res.set_xlabel('Streak Width (pixels)', fontsize=12)
        ax2_res.grid(True, alpha=0.3)
    else:
        width_bins = np.array([0, 1])
        mean_w, std_w, mean_w_um, std_w_um = np.nan, np.nan, np.nan, np.nan

    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Streak Width Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    ax2_t = ax2.twiny()
    x2lim = ax2.get_xlim()
    ax2_t.set_xlim(x2lim[0] / pixels_per_um, x2lim[1] / pixels_per_um)
    ax2_t.set_xlabel('Width (um)', fontsize=11, color='darkorange')
    ax2_t.tick_params(axis='x', labelcolor='darkorange')

    # Width -> radius -> volume -> mass
    mean_width_m = mean_w / pixels_per_m if np.isfinite(mean_w) else np.nan
    radius_m = (mean_width_m / 2.0) if np.isfinite(mean_width_m) else np.nan
    volume_m3 = (4.0/3.0) * np.pi * (radius_m**3) if np.isfinite(radius_m) else np.nan
    mass_kg = volume_m3 * density_kg_m3 if np.isfinite(volume_m3) else np.nan

    # ============================================================
    # 3) Speed histogram (Rayleigh fit) + residuals
    # ============================================================
    valid_len_mask = np.isfinite(lengths) & (lengths >= 0)
    v_mps = ((lengths[valid_len_mask] * px_to_m) / exposure_time_s).astype(float)

    sigma_fit, sigma_err, chi2_red_speed = np.nan, np.nan, np.nan
    v_mean, v_std, v_rms = np.nan, np.nan, np.nan
    rayleigh_mean_predicted = np.nan

    if v_mps.size > 0:
        vmin, vmax = float(np.min(v_mps)), float(np.max(v_mps))
        n_bins_speed = 30
        speed_bins_mps = np.linspace(vmin, vmax, n_bins_speed + 1) if vmax > vmin else np.array([vmin, vmin+1.0])
        bin_centers_v = (speed_bins_mps[:-1] + speed_bins_mps[1:]) / 2.0
        bin_width_v = speed_bins_mps[1] - speed_bins_mps[0] if len(speed_bins_mps) > 1 else 1.0

        hist_speed, _ = np.histogram(v_mps, bins=speed_bins_mps)
        n_total_v = v_mps.size

        sigma_fit, sigma_err = fit_rayleigh(v_mps)
        expected_speed = n_total_v * bin_width_v * rayleigh_2d_pdf(bin_centers_v, sigma_fit)

        residuals_v, chi2_red_speed, _ = compute_chi2_residuals(hist_speed, expected_speed, n_params=1)

        errors_v = np.sqrt(hist_speed)
        errors_v[errors_v == 0] = 1

        hist_density_v = hist_speed / (n_total_v * bin_width_v)
        errors_density_v = errors_v / (n_total_v * bin_width_v)

        # Step histogram
        ax3.stairs(hist_density_v, speed_bins_mps, color='seagreen', linewidth=2, label='Data')
        ax3.errorbar(bin_centers_v, hist_density_v, yerr=errors_density_v,
                     fmt='none', ecolor='seagreen', capsize=3, capthick=1.5, elinewidth=1.5)

        # Rayleigh fit (1000 points)
        x_fit = np.linspace(max(0, vmin), vmax, 1000)
        y_fit = rayleigh_2d_pdf(x_fit, sigma_fit)
        ax3.plot(x_fit, y_fit, 'r-', lw=2,
                 label=f'Rayleigh: $\\sigma$ = {sigma_fit*1e3:.3f} +/- {sigma_err*1e3:.3f} mm/s\n$\\chi^2_{{red}}$ = {chi2_red_speed:.2f}')

        v_mean = float(np.mean(v_mps))
        v_std  = float(np.std(v_mps))
        v_rms  = float(np.sqrt(np.mean(v_mps**2)))

        # Rayleigh predicted mean: <u> = sigma * sqrt(pi/2)
        rayleigh_mean_predicted = sigma_fit * np.sqrt(np.pi / 2.0)

        # Mean line (measured)
        ax3.axvline(v_mean, color='darkgreen', linestyle='--', linewidth=2,
                    label=f'Mean (data): {v_mean*1e3:.2f} +/- {v_std*1e3:.2f} mm/s')

        # Model predicted mean line
        ax3.axvline(rayleigh_mean_predicted, color='red', linestyle=':', linewidth=2,
                    label=f'Mean (Rayleigh): {rayleigh_mean_predicted*1e3:.2f} mm/s')

        ax3.legend(fontsize=9)
        ax3.set_xlim(vmin, vmax if vmax > vmin else vmin + 1.0)
        ax3.set_ylim(0, np.max(hist_density_v) * 1.3)

        # Residuals subplot
        ax3_res.bar(bin_centers_v, residuals_v, width=0.8*bin_width_v, color='seagreen', alpha=0.7, edgecolor='black')
        ax3_res.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3_res.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3_res.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3_res.set_xlim(ax3.get_xlim())
        ax3_res.set_ylabel('Residuals\n$(O-E)/\\sqrt{E}$', fontsize=10)
        ax3_res.set_xlabel('Speed (m/s)', fontsize=12)
        ax3_res.grid(True, alpha=0.3)
    else:
        speed_bins_mps = np.array([0.0, 1.0])

    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Projected Speed (2D Rayleigh)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])

    ax3_t = ax3.twiny()
    x3lim = ax3.get_xlim()
    ax3_t.set_xlim(x3lim[0] * 1e3, x3lim[1] * 1e3)
    ax3_t.set_xlabel('Speed (mm/s)', fontsize=11, color='darkgreen')
    ax3_t.tick_params(axis='x', labelcolor='darkgreen')

    # Title and save/show
    total_droplets = int(lengths.size)
    fig.suptitle(
        f'Independent Analysis - {frames_analyzed} frames, {total_droplets} droplets',
        fontsize=15, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[save] Figure saved to: {save_path}")
    else:
        plt.show()

    # Assemble results
    results = {
        "metadata_used": {
            "pixels_per_um": pixels_per_um,
            "fps": fps,
            "exposure_time_s": exposure_time_s,
            "beam_waist_px": beam_waist_px,
            "frames_analyzed": frames_analyzed,
            "beam_area_mm2": beam_cross_section_mm2,
            "density_kg_m3": density_kg_m3,
            "volume_mm3": volume_mm3,
            "volume_source": volume_source,
            "r_used_mm": r_used_mm,
            "z_used_mm": z_used_mm
        },
        "counts": {
            "lambda_fit": lam,
            "lambda_fit_err": lam_err,
            "mean_per_frame": mean_count,
            "std_per_frame": std_count,
            "mean_density_per_mm2": mean_count_per_mm2,
            "std_density_per_mm2": std_count_per_mm2,
            "mean_density_per_mm3": mean_density_per_mm3,
            "std_density_per_mm3": std_density_per_mm3,
            "chi2_reduced": chi2_red_counts,
            "hist_bins": count_bins
        },
        "width": {
            "hist_bins": width_bins,
            "lambda_fit": lambda_width,
            "lambda_fit_err": lambda_width_err,
            "mean_width_px": mean_w,
            "std_width_px": std_w,
            "mean_width_um": mean_w_um,
            "std_width_um": std_w_um,
            "radius_m": radius_m,
            "volume_m3": volume_m3,
            "mass_kg": mass_kg,
            "chi2_reduced": chi2_red_width
        },
        "speed": {
            "speed_bins_mps": speed_bins_mps,
            "v_mean_mps": v_mean,
            "v_std_mps": v_std,
            "v_rms_mps": v_rms,
            "sigma_fit_mps": sigma_fit,
            "sigma_fit_err_mps": sigma_err,
            "rayleigh_mean_predicted_mps": rayleigh_mean_predicted,
            "chi2_reduced": chi2_red_speed
        }
    }
    return results


# =============================================================================
# STEP 2: Combined analysis (scaffold)
# =============================================================================

def analyze_combined(
    step1_results: Dict[str, Any],
    per_droplet: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Placeholder for future joint analysis."""
    w = step1_results.get("width", {})
    s = step1_results.get("speed", {})
    resolved_mass  = w.get("mass_kg", np.nan)
    resolved_sigma = s.get("sigma_fit_mps", np.nan)

    combined = {
        "resolved_mass_kg": resolved_mass,
        "resolved_sigma_mps": resolved_sigma,
        "notes": "Combined analysis placeholder: currently returns width->mass and Rayleigh sigma."
    }

    return {
        "independent": step1_results,
        "combined": combined
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data_file = "droplet_data.npy"
    results_json = "test_results.json"
    save_plot_step1 = None

    print(f"[load] Loading droplet data from: {data_file}")
    per_droplet = load_droplet_data(data_file, results_json=results_json)

    step1 = analyze_independent(
        per_droplet,
        density_kg_m3=3520.0,
        save_path=save_plot_step1
    )

    step2 = analyze_combined(step1, per_droplet)

    # Console summary
    meta = step1["metadata_used"]
    counts_info = step1["counts"]
    width_info  = step1["width"]
    speed_info  = step1["speed"]

    frames_analyzed = meta.get('frames_analyzed', 0)
    exposure_time_s = meta.get('exposure_time_s', np.nan)
    fps             = meta.get('fps', np.nan)
    beam_area_mm2   = meta.get('beam_area_mm2', np.nan)
    volume_mm3      = meta.get('volume_mm3', np.nan)

    mean_droplets = counts_info.get("mean_per_frame", np.nan)
    std_droplets  = counts_info.get("std_per_frame", np.nan)
    chi2_counts   = counts_info.get("chi2_reduced", np.nan)

    dens_mm2_mean = counts_info.get("mean_density_per_mm2", np.nan)
    dens_mm2_std  = counts_info.get("std_density_per_mm2", np.nan)
    dens_mm3_mean = counts_info.get("mean_density_per_mm3", np.nan)
    dens_mm3_std  = counts_info.get("std_density_per_mm3", np.nan)

    mean_width_px = width_info.get("mean_width_px", np.nan)
    std_width_px  = width_info.get("std_width_px", np.nan)
    mean_width_um = width_info.get("mean_width_um", np.nan)
    std_width_um  = width_info.get("std_width_um", np.nan)
    chi2_width    = width_info.get("chi2_reduced", np.nan)

    v_mean_mps = speed_info.get("v_mean_mps", np.nan)
    v_std_mps  = speed_info.get("v_std_mps", np.nan)
    sigma_mps  = speed_info.get("sigma_fit_mps", np.nan)
    chi2_speed = speed_info.get("chi2_reduced", np.nan)
    rayleigh_mean = speed_info.get("rayleigh_mean_predicted_mps", np.nan)

    v_mean_mmps = v_mean_mps * 1e3 if np.isfinite(v_mean_mps) else np.nan
    v_std_mmps  = v_std_mps  * 1e3 if np.isfinite(v_std_mps)  else np.nan

    print("\n" + "="*62)
    print("PIPELINE SUMMARY")
    print("="*62)
    print(f"Frames analyzed: {frames_analyzed}")
    if np.isfinite(exposure_time_s) and np.isfinite(fps):
        print(f"Exposure time:  {exposure_time_s:.6f} s  (fps={fps:.2f})")
    if np.isfinite(beam_area_mm2):
        print(f"Beam area:      {beam_area_mm2:.4f} mm^2")
    if np.isfinite(volume_mm3):
        print(f"Volume factor:  {volume_mm3:.6f} mm^3  (pi r^2 z)")

    print("\n--- Step 1 (Independent) ---")
    if np.isfinite(mean_droplets) and np.isfinite(std_droplets):
        chi2_str = f"  (chi2_red={chi2_counts:.2f})" if np.isfinite(chi2_counts) else ""
        print(f"Mean droplets per frame: {mean_droplets:.2f} +/- {std_droplets:.2f}{chi2_str}")
    else:
        print("Mean droplets per frame: n/a")

    if np.isfinite(dens_mm2_mean) and np.isfinite(dens_mm2_std):
        print(f"Areal density:           {dens_mm2_mean:.4f} +/- {dens_mm2_std:.4f}  droplets / mm^2")
    else:
        print("Areal density:           n/a")

    if np.isfinite(dens_mm3_mean) and np.isfinite(dens_mm3_std):
        print(f"Volumetric density:      {dens_mm3_mean:.6f} +/- {dens_mm3_std:.6f}  droplets / mm^3")
    else:
        print("Volumetric density:      n/a")

    if np.isfinite(mean_width_px) and np.isfinite(std_width_px):
        chi2_str = f"  (chi2_red={chi2_width:.2f})" if np.isfinite(chi2_width) else ""
        if np.isfinite(mean_width_um) and np.isfinite(std_width_um):
            print(f"Average width:           {mean_width_px:.2f} +/- {std_width_px:.2f} px  "
                f"({mean_width_um:.2f} +/- {std_width_um:.2f} um){chi2_str}")
        else:
            print(f"Average width:           {mean_width_px:.2f} +/- {std_width_px:.2f} px{chi2_str}")
    else:
        print("Average width:           n/a")

    if np.isfinite(v_mean_mps) and np.isfinite(v_std_mps):
        chi2_str = f"  (chi2_red={chi2_speed:.2f})" if np.isfinite(chi2_speed) else ""
        print(f"Average speed (data):    {v_mean_mps:.6f} +/- {v_std_mps:.6f} m/s  "
            f"({v_mean_mmps:.2f} +/- {v_std_mmps:.2f} mm/s){chi2_str}")
    elif np.isfinite(v_mean_mps):
        print(f"Average speed (data):    {v_mean_mps:.6f} m/s")
    else:
        print("Average speed (data):    n/a")

    if np.isfinite(sigma_mps):
        print(f"Rayleigh sigma:          {sigma_mps*1e3:.3f} mm/s")
    if np.isfinite(rayleigh_mean):
        print(f"Rayleigh predicted mean: {rayleigh_mean*1e3:.2f} mm/s")

    print("\n--- Step 2 (Combined - placeholder) ---")
    resolved_mass  = step2['combined'].get('resolved_mass_kg', np.nan)
    resolved_sigma = step2['combined'].get('resolved_sigma_mps', np.nan)
    print(f"Resolved mass:           {resolved_mass:.3e} kg" if np.isfinite(resolved_mass) else "Resolved mass: n/a")
    print(f"Resolved sigma:          {resolved_sigma*1e3:.3f} mm/s" if np.isfinite(resolved_sigma) else "Resolved sigma: n/a")
    print("="*62 + "\n")
