"""
Environmental Correlations

Loads a longitudinal analysis numpy file, determines the start/end time of the
analysis window, queries the InfluxDB database for environmental sensor data
(RTDs and pressure gauges) over that window, then produces scatter plots of
temperature/pressure vs each longitudinal parameter.

Sensor mapping
--------------
  RTD_1          → TEMP_CUBE   (K)
  RTD_2          → TEMP_BASE   (K)
  RTD_3          → TEMP_TOP    (K)
  RTD_4          → TEMP_BOT    (K)
  Setra 225 - Fill Line    → PRESSURE_CUBE   (units from sensor)
  Setra 225 - Xenon Bottle → PRESSURE_BOTTLE (units from sensor)

Usage: edit the USER SETTINGS block in the __main__ section and run:
    python environmental_correlations.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from influxdb_client import InfluxDBClient
import pytz


# =============================================================================
# InfluxDB connection parameters (shared with query_db.py)
# =============================================================================

INFLUX_URL    = "http://gl-sft1200.stdusr.yale.internal:2504"
INFLUX_ORG    = "xbox-server"
INFLUX_TOKEN  = "73DWRw1C9hwf0U1bqz7VqtWAVt7tINrpTpH3Gmas756vmHwd_lyAjOrr7GtLZqXvwQ1-d3iGwev5cL4WtLPbXw=="
INFLUX_BUCKET_CRYO = "Cryostat"
INFLUX_BUCKET_GAS  = "Gas Handling System"
DOWNSAMPLE_EVERY   = "10s"


# =============================================================================
# Helpers
# =============================================================================

def _get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _utc_timestamp_to_rfc3339(ts: float) -> str:
    """Convert a UTC Unix timestamp to an RFC-3339 string for Flux queries."""
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_longitudinal_data(npy_path: str) -> np.ndarray:
    """Load a longitudinal analysis numpy file."""
    data = np.load(npy_path, allow_pickle=False)
    print(f"[env_corr] Loaded longitudinal data: {npy_path}")
    print(f"           {len(data)} chunks, fields: {data.dtype.names}")
    return data


def get_time_range_from_data(data: np.ndarray):
    """
    Return (start_utc, end_utc) from a longitudinal data array.

    Uses start_utc stored in the array plus the chunk time span to determine
    the wall-clock time range.  If start_utc is NaN for all rows, raises an
    error — you must set START_UTC when running longitudinal_analysis.py.
    """
    start_utc_vals = data['start_utc']
    valid = start_utc_vals[np.isfinite(start_utc_vals)]
    if len(valid) == 0:
        raise ValueError(
            "No valid start_utc in longitudinal data. "
            "Re-run longitudinal_analysis.py with START_UTC set."
        )
    start_utc = float(valid[0])
    # Total time span of the analysis window
    max_chunk_end_s = float(np.nanmax(data['chunk_end_time_s']))
    # chunk times are relative to the start of the video analysis window.
    # The absolute UTC end time is start_utc + max_chunk_end_s.
    end_utc = start_utc + max_chunk_end_s
    return start_utc, end_utc


def query_environmental_data(start_utc: float, end_utc: float) -> pd.DataFrame:
    """
    Query RTD and pressure data from InfluxDB for the given UTC time window.

    Returns a DataFrame indexed by UTC datetime with columns:
      TEMP_CUBE, TEMP_BASE, TEMP_TOP, TEMP_BOT, PRESSURE_CUBE, PRESSURE_BOTTLE
    All temperatures are in Kelvin (converted from Celsius).
    """
    start_rfc = _utc_timestamp_to_rfc3339(start_utc)
    end_rfc   = _utc_timestamp_to_rfc3339(end_utc)

    # RTD fields in the Cryostat bucket (stored as _C — degrees Celsius)
    rtd_fields = ["RTD1_K", "RTD2_K", "RTD3_K", "RTD4_K"]
    # Try both _K and _C suffixes — the database may store either
    rtd_fields_c = ["RTD1_C", "RTD2_C", "RTD3_C", "RTD4_C"]
    rtd_fields_k = ["RTD1_K", "RTD2_K", "RTD3_K", "RTD4_K"]

    flux_rtd_c = f"""
from(bucket: "{INFLUX_BUCKET_CRYO}")
  |> range(start: {start_rfc}, stop: {end_rfc})
  |> filter(fn: (r) => r["_measurement"] == "RTD")
  |> filter(fn: (r) => r["_field"] == "RTD1_C" or r["_field"] == "RTD2_C"
                    or r["_field"] == "RTD3_C" or r["_field"] == "RTD4_C")
  |> aggregateWindow(every: {DOWNSAMPLE_EVERY}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "RTD1_C", "RTD2_C", "RTD3_C", "RTD4_C"])
  |> sort(columns: ["_time"])
"""

    flux_fill_line = f"""
from(bucket: "{INFLUX_BUCKET_GAS}")
  |> range(start: {start_rfc}, stop: {end_rfc})
  |> filter(fn: (r) => r["_measurement"] == "Setra 225 - Fill Line")
  |> aggregateWindow(every: {DOWNSAMPLE_EVERY}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""

    flux_xenon_bottle = f"""
from(bucket: "{INFLUX_BUCKET_GAS}")
  |> range(start: {start_rfc}, stop: {end_rfc})
  |> filter(fn: (r) => r["_measurement"] == "Setra 225 - Xenon Bottle")
  |> aggregateWindow(every: {DOWNSAMPLE_EVERY}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""

    print(f"[env_corr] Querying InfluxDB from {start_rfc} to {end_rfc} ...")

    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=60_000) as client:
        api = client.query_api()

        df_rtd = api.query_data_frame(query=flux_rtd_c, org=INFLUX_ORG)
        df_fill = api.query_data_frame(query=flux_fill_line, org=INFLUX_ORG)
        df_xen  = api.query_data_frame(query=flux_xenon_bottle, org=INFLUX_ORG)

    # Coerce list results
    if isinstance(df_rtd, list):
        df_rtd = pd.concat(df_rtd, ignore_index=True) if df_rtd else pd.DataFrame()
    if isinstance(df_fill, list):
        df_fill = pd.concat(df_fill, ignore_index=True) if df_fill else pd.DataFrame()
    if isinstance(df_xen, list):
        df_xen = pd.concat(df_xen, ignore_index=True) if df_xen else pd.DataFrame()

    def _prep(df, time_col="_time"):
        df = df.rename(columns={time_col: "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        return df.drop(columns=["result", "table", "_start", "_stop"], errors="ignore")

    df_rtd  = _prep(df_rtd)
    df_fill = _prep(df_fill)
    df_xen  = _prep(df_xen)

    # Convert RTD from Celsius to Kelvin
    for col_c in ["RTD1_C", "RTD2_C", "RTD3_C", "RTD4_C"]:
        if col_c in df_rtd.columns:
            df_rtd[col_c] = df_rtd[col_c] + 273.15

    # Rename columns
    rename_rtd = {"RTD1_C": "TEMP_CUBE", "RTD2_C": "TEMP_BASE",
                  "RTD3_C": "TEMP_TOP",  "RTD4_C": "TEMP_BOT"}
    df_rtd = df_rtd.rename(columns=rename_rtd)

    # Pressure: keep the first numeric column
    def _first_numeric(df, new_name):
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return pd.DataFrame()
        return df[[num_cols[0]]].rename(columns={num_cols[0]: new_name})

    df_fill = _first_numeric(df_fill, "PRESSURE_CUBE")
    df_xen  = _first_numeric(df_xen,  "PRESSURE_BOTTLE")

    # Merge all onto RTD timestamps
    env = df_rtd[["TEMP_CUBE", "TEMP_BASE", "TEMP_TOP", "TEMP_BOT"]].copy()
    for df_extra in [df_fill, df_xen]:
        if not df_extra.empty:
            env = pd.merge_asof(env.sort_index(), df_extra.sort_index(),
                                left_index=True, right_index=True,
                                direction="nearest")

    print(f"[env_corr] Environmental data: {len(env)} rows, columns: {list(env.columns)}")
    return env


def interpolate_env_to_chunks(env_df: pd.DataFrame, data: np.ndarray,
                               start_utc: float) -> pd.DataFrame:
    """
    For each chunk mid-time, interpolate environmental data to that timestamp.
    Returns a DataFrame aligned to the chunk array.
    """
    chunk_utc_times = start_utc + data['chunk_mid_time_s']
    chunk_datetimes = pd.to_datetime(chunk_utc_times, unit='s', utc=True)

    result = {}
    for col in env_df.columns:
        series = env_df[col].dropna()
        if series.empty:
            result[col] = np.full(len(data), np.nan)
            continue
        # Interpolate by timestamp
        ts_env    = series.index.astype(np.int64) / 1e9  # seconds
        ts_chunks = chunk_datetimes.astype(np.int64) / 1e9
        result[col] = np.interp(ts_chunks, ts_env, series.values,
                                left=np.nan, right=np.nan)

    return pd.DataFrame(result)


def make_scatter_plots(data: np.ndarray, env_aligned: pd.DataFrame,
                        plots_dir: str, basename: str):
    """
    Create scatter plots of environmental variables vs longitudinal parameters.
    Saves to plots/environmental_correlations/<basename>_<env>_vs_<param>.png
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Longitudinal parameters to correlate
    long_params = {
        'mean_density_mm3':  'Density (droplets/mm³)',
        'mean_count':        'Mean droplets/frame',
        'mean_width_um':     'Mean width (µm)',
        'mean_speed_mps':    'Mean speed (m/s)',
        'mean_vy_mps':       'Mean |v_r| (m/s)',
        'mean_vz_mps':       'Mean |v_z| (m/s)',
        'wind_direction_deg':'Wind direction (°)',
        'anisotropy_ratio':  'Anisotropy ratio',
    }

    temp_params = [c for c in ['TEMP_CUBE', 'TEMP_BASE', 'TEMP_TOP', 'TEMP_BOT']
                   if c in env_aligned.columns]
    pres_params = [c for c in ['PRESSURE_CUBE', 'PRESSURE_BOTTLE']
                   if c in env_aligned.columns]

    temp_labels = {
        'TEMP_CUBE':   'T_CUBE (K)',
        'TEMP_BASE':   'T_BASE (K)',
        'TEMP_TOP':    'T_TOP (K)',
        'TEMP_BOT':    'T_BOT (K)',
    }
    pres_labels = {
        'PRESSURE_CUBE':   'P_CUBE (sensor units)',
        'PRESSURE_BOTTLE': 'P_BOTTLE (sensor units)',
    }

    for env_col, env_params, env_label_map in [
        ('temperature', temp_params, temp_labels),
        ('pressure',    pres_params, pres_labels),
    ]:
        for env_var in env_params:
            env_vals = env_aligned[env_var].values
            env_label = env_label_map.get(env_var, env_var)

            valid_env = np.isfinite(env_vals)
            if not np.any(valid_env):
                print(f"  [skip] no valid data for {env_var}")
                continue

            n_params = len(long_params)
            ncols = 2
            nrows = int(np.ceil(n_params / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
            axes = np.array(axes).ravel()

            for ax_i, (param_key, param_label) in enumerate(long_params.items()):
                ax = axes[ax_i]
                if param_key not in data.dtype.names:
                    ax.set_visible(False)
                    continue
                y_vals = data[param_key].astype(float)
                valid = valid_env & np.isfinite(y_vals)
                if np.sum(valid) < 2:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                            transform=ax.transAxes)
                    ax.set_title(param_label, fontsize=10)
                    continue

                ax.scatter(env_vals[valid], y_vals[valid],
                           c=data['chunk_mid_time_s'][valid],
                           cmap='viridis', s=30, alpha=0.8, edgecolors='none')
                ax.set_xlabel(env_label, fontsize=10)
                ax.set_ylabel(param_label, fontsize=10)
                ax.set_title(f"{param_label} vs {env_var}", fontsize=10)
                ax.grid(True, alpha=0.3)

            # Hide unused axes
            for ax_i in range(n_params, len(axes)):
                axes[ax_i].set_visible(False)

            # Colorbar (time axis)
            sm = plt.cm.ScalarMappable(
                cmap='viridis',
                norm=plt.Normalize(vmin=np.nanmin(data['chunk_mid_time_s']),
                                   vmax=np.nanmax(data['chunk_mid_time_s']))
            )
            sm.set_array([])
            fig.colorbar(sm, ax=axes[:n_params], label='Chunk mid-time (s)',
                         shrink=0.6, pad=0.01)

            fig.suptitle(f"{basename}: {env_label} vs longitudinal parameters",
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            out_name = f"{basename}_{env_var}_vs_longitudinal.png"
            out_path = os.path.join(plots_dir, out_name)
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  [env_corr] saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # USER SETTINGS — edit these
    # =========================================================================

    # Path to the longitudinal analysis numpy file
    # (produced by longitudinal_analysis.py)
    LONGITUDINAL_FILE = os.path.join(
        _get_project_root(), "data", "water_constexp2",
        "longitudinal_analysis_water_constexp2_2026-03-11_09-16-43.npy"
    )

    # =========================================================================

    root_dir  = _get_project_root()
    data      = load_longitudinal_data(LONGITUDINAL_FILE)
    basename  = os.path.basename(LONGITUDINAL_FILE)
    # Extract video basename from filename: longitudinal_analysis_<basename>_<timestamp>.npy
    parts = basename.replace("longitudinal_analysis_", "").rsplit("_", 3)
    video_basename = parts[0] if parts else "unknown"

    # Get analysis time range
    start_utc, end_utc = get_time_range_from_data(data)
    print(f"[env_corr] Analysis window: "
          f"{datetime.datetime.fromtimestamp(start_utc, tz=datetime.timezone.utc).isoformat()} "
          f"→ "
          f"{datetime.datetime.fromtimestamp(end_utc, tz=datetime.timezone.utc).isoformat()}")

    # Query environmental data
    env_df = query_environmental_data(start_utc, end_utc)

    # Interpolate environmental data to chunk mid-times
    env_aligned = interpolate_env_to_chunks(env_df, data, start_utc)

    # Make scatter plots
    plots_dir = os.path.join(root_dir, "plots", "environmental_correlations")
    make_scatter_plots(data, env_aligned, plots_dir, video_basename)

    print(f"\n[env_corr] Done. Plots saved to: {plots_dir}")
