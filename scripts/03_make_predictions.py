import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data_processed"
MODEL_DIR = BASE_DIR / "models"

def main():
    # 1) Load model
    model_path = MODEL_DIR / "xgb_water_model.pkl"
    model: XGBRegressor = joblib.load(model_path)

    # 2) Load feature dataset (your 23 KB file)
    feat_path = PROC_DIR / "water_daily_with_features.csv"
    df_feat = pd.read_csv(feat_path, parse_dates=["date"])

    feature_cols = [
        "usage_lag_1", "usage_lag_2", "usage_lag_7",
        "rolling_7_mean", "rolling_7_std",
        "day_of_week", "month", "is_weekend",
        "is_holiday", "is_semester", "is_exam",
        "capacity"
    ]

    X = df_feat[feature_cols]

    # 3) Predict next-day usage for each row
    df_feat["predicted_next_day"] = model.predict(X)

    # Align predictions to the target date (next day)
    df_feat["target_date"] = df_feat["date"] + pd.Timedelta(days=1)

    # 4) Compact table for Tableau
    pred = df_feat[["campus_id", "name", "target_date", "target_next_day", "predicted_next_day"]].copy()
    pred.rename(columns={
        "name": "campus_name",
        "target_date": "date",
        "target_next_day": "actual_usage_kl",
        "predicted_next_day": "predicted_usage_kl"
    }, inplace=True)

    out_path = PROC_DIR / "predictions_daily.csv"
    pred.to_csv(out_path, index=False)
    print("✅ Saved predictions to:", out_path)

if __name__ == "__main__":
    main()
