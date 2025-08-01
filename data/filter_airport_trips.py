import pandas as pd
from pathlib import Path

# Config
PARQUET_PATH = "data/yellow_tripdata_2025-02.parquet"
OUTPUT_DIR = Path("data/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AIRPORT_IDS = {1, 132, 138}  # JFK, LGA, EWR

# ---------------------------
# 1. Load and filter airport trips
# ---------------------------
print("Reading Parquet...")
df = pd.read_parquet(PARQUET_PATH)
df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"])

print("Filtering airport-related trips...")
airport_df = df[
    df["PULocationID"].isin(AIRPORT_IDS) |
    df["DOLocationID"].isin(AIRPORT_IDS)
].copy()

# ---------------------------
# 2. Training: Feb 1–12, stratified 2%
# ---------------------------
print("Generating training set from Feb 1–12...")
train_df = airport_df[
    airport_df["pickup_dt"].dt.date.between(
        pd.to_datetime("2025-02-01").date(),
        pd.to_datetime("2025-02-07").date()
    )
]

train_sample = (
    train_df
    .groupby(train_df["pickup_dt"].dt.date, group_keys=False)
    .apply(lambda x: x.sample(frac=0.1, random_state=2425))
    .reset_index(drop=True)
)

train_sample.to_csv(OUTPUT_DIR / "train_airport_sample_2025-02.csv", index=False)
print(f"Saved training set: {len(train_sample):,} rows")

# ---------------------------
# 3. Testing: One-minute windows
# ---------------------------
def extract_test_window(df, start_str, label):
    start = pd.to_datetime(start_str)
    end = start + pd.Timedelta(minutes=1)
    sliced = df[(df["pickup_dt"] >= start) & (df["pickup_dt"] < end)].copy()
    out_path = OUTPUT_DIR / f"test_{label}.csv"
    sliced.to_csv(out_path, index=False)
    print(f"Saved test slice '{label}': {len(sliced)} rows")

print("Extracting 1-minute test windows...")

# Peak hour – Monday morning commute
extract_test_window(airport_df, "2025-02-24 08:00:00", "peak_2025-02-24_0800")

# Midday off-peak – Tuesday lunch hour
extract_test_window(airport_df, "2025-02-25 13:00:00", "offpeak_2025-02-25_1300")

# Late night – Valentine's Day
extract_test_window(airport_df, "2025-02-14 23:00:00", "latenight_2025-02-14_2300")
