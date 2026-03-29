import pandas as pd
from pathlib import Path

# Go up from scripts/ to project root
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
PROC_DIR = BASE_DIR / "data_processed"
PROC_DIR.mkdir(exist_ok=True)


def main():
    # 1) Load raw water data
    water_path = RAW_DIR / "water_consumption.csv"
    df_water = pd.read_csv(water_path)

    # -------------------------------
    # FIX 1: Safe timestamp parsing
    # -------------------------------
    df_water["timestamp"] = pd.to_datetime(
    df_water["timestamp"]
        .astype(str)
        .str.strip()
        .str.replace(".", ":", regex=False),  # 🔥 FIX HERE
    format="%d-%m-%Y %H:%M",
    errors="coerce"
    )
    print("Invalid timestamps:", df_water["timestamp"].isna().sum())

    # Drop invalid timestamps
    df_water = df_water.dropna(subset=["timestamp"])

    # Extract date
    df_water["date"] = df_water["timestamp"].dt.date

    # -------------------------------
    # 2) Aggregate daily usage
    # -------------------------------
    daily = (
        df_water
        .groupby(["campus_id", "date"], as_index=False)["consumption"]
        .sum()
    )
    daily.rename(columns={"consumption": "usage_kl"}, inplace=True)

    # -------------------------------
    # 3) Load campus metadata
    # -------------------------------
    campus_path = RAW_DIR / "campus_meta.csv"
    df_campus = pd.read_csv(campus_path)

    daily = daily.merge(df_campus, left_on="campus_id", right_on="id", how="left")

    # -------------------------------
    # 4) Load calendar data
    # -------------------------------
    cal_path = RAW_DIR / "calender.csv"
    df_cal = pd.read_csv(cal_path)

    # Ensure first column is named 'date'
    if "date" not in df_cal.columns:
        df_cal.rename(columns={df_cal.columns[0]: "date"}, inplace=True)

    # -------------------------------
    # FIX 2: Remove invalid rows
    # -------------------------------
    df_cal = df_cal.dropna(subset=["date"])
    df_cal = df_cal[df_cal["date"] != "date"]  # remove accidental header rows

    # -------------------------------
    # FIX 3: Robust date parsing
    # -------------------------------
    df_cal["date"] = pd.to_datetime(
        df_cal["date"],
        dayfirst=True,       # handles 01-01-2016
        errors="coerce"
    )

    # Drop invalid parsed dates
    df_cal = df_cal.dropna(subset=["date"])
    df_cal["date"] = df_cal["date"].dt.date

    # -------------------------------
    # Rename columns if needed
    # -------------------------------
    if len(df_cal.columns) >= 4:
        df_cal.columns = ["date", "is_holiday", "is_semester", "is_exam"]

    # -------------------------------
    # 5) Merge calendar data
    # -------------------------------
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily.merge(df_cal, on="date", how="left")

    # -------------------------------
    # 6) Save processed data
    # -------------------------------
    out_path = PROC_DIR / "water_daily_features.csv"
    daily.to_csv(out_path, index=False)

    print("✅ Saved successfully:", out_path)


if __name__ == "__main__":
    main()