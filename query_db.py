import matplotlib.pyplot as plt
import numpy as np
from influxdb_client import InfluxDBClient
import pandas as pd
import matplotlib.dates as mdates
import pytz

URL   = "http://gl-sft1200.stdusr.yale.internal:2504"     # e.g. https://us-east-1-1.aws.cloud2.influxdata.com
ORG   = "xbox-server"                     # org name or ID
TOKEN = "73DWRw1C9hwf0U1bqz7VqtWAVt7tINrpTpH3Gmas756vmHwd_lyAjOrr7GtLZqXvwQ1-d3iGwev5cL4WtLPbXw=="                   # read token
BUCKET= "Cryostat"                  # bucket name

TIME_RANGE = '-8d'  
DOWNSAMPLE_EVERY  = "10s" 
FIELDS = [
    "RTD1_C", "RTD2_C", "RTD3_C", "RTD4_C",
    "PID1_SP", "PID2_SP",
    "PID1_Output", "PID2_Output",
    "PID1_PV", "PID2_PV",
]

flux = f"""
from(bucket: "{BUCKET}")
  |> range(start: {TIME_RANGE})
  |> filter(fn: (r) => r["_measurement"] == "RTD" or r["_measurement"] == "PLC_PID1" or r["_measurement"] == "PLC_PID2")
  |> filter(fn: (r) => r["_field"] == "RTD2_C" or r["_field"] == "RTD3_C" or r["_field"] == "RTD4_C" or r["_field"] == "RTD1_C" or r["_field"] == "PID1_SP" or r["_field"] == "PID2_SP" or r["_field"] == "PID2_Output" or r["_field"] == "PID1_Output" or r["_field"] == "PID1_PV" or r["_field"] == "PID2_PV")
  |> aggregateWindow(every: {DOWNSAMPLE_EVERY}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time",{','.join(f'"{f}"' for f in FIELDS)}])
  |> sort(columns: ["_time"])
"""

with InfluxDBClient(url=URL, token=TOKEN, org=ORG, timeout=60_000) as client:
    api = client.query_api()

    # --- Cryostat data (RTD + PID) ---
    df = api.query_data_frame(query=flux, org=ORG)

    # --- Pressure data (Gas Handling System) ---
    flux_pres = f"""
from(bucket: "Gas Handling System")
  |> range(start: {TIME_RANGE})
  |> filter(fn: (r) => r["_measurement"] == "Setra 225 - Fill Line")
  |> aggregateWindow(every: {DOWNSAMPLE_EVERY}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
    df_pres = api.query_data_frame(query=flux_pres, org=ORG)

if isinstance(df, list):
    df = pd.concat(df, ignore_index=True)
if isinstance(df_pres, list):
    df_pres = pd.concat(df_pres, ignore_index=True)

df = df.rename(columns={"_time": "time"}).set_index("time")
df.index = pd.to_datetime(df.index, utc=True)
df = df.drop(columns=["result", "table"], errors="ignore")

df_pres = df_pres.rename(columns={"_time": "time"}).set_index("time")
df_pres.index = pd.to_datetime(df_pres.index, utc=True)
df_pres = df_pres.drop(columns=["result", "table", "_start", "_stop"], errors="ignore")
# Rename pressure column(s) to a single "PRES" column
# Keep only numeric columns from pressure data
pres_cols = df_pres.select_dtypes(include="number").columns.tolist()
if len(pres_cols) == 1:
    df_pres = df_pres.rename(columns={pres_cols[0]: "PRES"})
else:
    # If multiple fields, take the first numeric one
    df_pres = df_pres[pres_cols].rename(columns={pres_cols[0]: "PRES"})
df_pres = df_pres[["PRES"]]

# Merge pressure onto Cryostat timestamps (nearest match, no extra rows)
df = df.sort_index()
df_pres = df_pres.sort_index()
df = pd.merge_asof(df, df_pres, left_index=True, right_index=True, direction="nearest")

df.to_csv("Cryostat_RTD_PID_downsampled.csv")
print("Saved: Cryostat_RTD_PID_downsampled.csv")