import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data_processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily data with columns at least:
        campus_id, date, usage_kl, is_holiday, is_semester, is_exam, capacity
    Output: same + lag/rolling/calendar features + target_next_day
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    all_rows = []

    # Build features per campus so lags/rolling are not mixed across campuses
    for campus_id, grp in df.groupby("campus_id"):
        g = grp.sort_values("date").copy()

        # Lag features
        g["usage_lag_1"] = g["usage_kl"].shift(1)
        g["usage_lag_2"] = g["usage_kl"].shift(2)
        g["usage_lag_7"] = g["usage_kl"].shift(7)

        # Rolling stats
        g["rolling_7_mean"] = g["usage_kl"].rolling(window=7).mean()
        g["rolling_7_std"] = g["usage_kl"].rolling(window=7).std()

        # Calendar features
        g["day_of_week"] = g["date"].dt.dayofweek
        g["month"] = g["date"].dt.month
        g["is_weekend"] = g["day_of_week"].isin([5, 6]).astype(int)

        # Target: next-day usage
        g["target_next_day"] = g["usage_kl"].shift(-1)

        all_rows.append(g)

    df_feat = pd.concat(all_rows, ignore_index=True)

    # Drop rows where we can't compute lags/target
    df_feat = df_feat.dropna(
        subset=["usage_lag_1", "usage_lag_7", "rolling_7_mean", "target_next_day"]
    )

    return df_feat


def main():
    # 1) Load processed daily data
    data_path = PROC_DIR / "water_daily_features.csv"
    df = pd.read_csv(data_path)

    # If calendar columns are numeric but read as float, fill NaNs with 0
    for col in ["is_holiday", "is_semester", "is_exam"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2) Create ML features
    df_feat = create_features(df)

    feature_cols = [
        "usage_lag_1", "usage_lag_2", "usage_lag_7",
        "rolling_7_mean", "rolling_7_std",
        "day_of_week", "month", "is_weekend",
        "is_holiday", "is_semester", "is_exam",
        "capacity"
    ]

    X = df_feat[feature_cols]
    y = df_feat["target_next_day"]

    # 3) Time-based split (80% train, 20% test)
    df_feat = df_feat.sort_values("date")
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4) Train XGBoost model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("MAE:", mae)
    print("RMSE:", rmse)

    # 6) Save model and feature dataset
    model_path = MODEL_DIR / "xgb_water_model.pkl"
    joblib.dump(model, model_path)
    print("✅ Model saved to", model_path)

    out_feat = PROC_DIR / "water_daily_with_features.csv"
    df_feat.to_csv(out_feat, index=False)
    print("✅ Feature dataset saved to", out_feat)


if __name__ == "__main__":
    main()
