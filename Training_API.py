# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 11:56:05 2025

@author: Anirban Boral
"""

"""
Created on Sun Nov 09 2025

Author: Anirban Boral
Purpose: One-shot automated training from data.gov.in API (last 7 years), no daily updates.
"""

import os
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

# ============ CONFIG ============
API_BASE = "https://www.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity#api"  # <- put AGMARKNET resource id
API_KEY = os.getenv("AGMARKNET_API_KEY", "579b464db66ec23bdd000001ddd500aa4d01448342adaeed0fb55618")  # replace with your real key for full results
FORMAT = "json"     # json | csv | xml
LIMIT = 1000        # sample key returns only 10; real key can use 1000
SLEEP_SEC = 0.15    # polite delay between pages
OUT_MODELS_DIR = "Models"
OUT_OUTPUTS_DIR = "Outputs"
os.makedirs(OUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_OUTPUTS_DIR, exist_ok=True)

# Optional scoping (you can leave None to fetch all India)
STATE = None         # e.g., "Maharashtra"
DISTRICT = None      # e.g., "Nashik"
COMMODITY = None     # e.g., "Onion"

# Pull last N years (inclusive)
YEARS_BACK = 7
START_DATE = (datetime.now() - timedelta(days=365*YEARS_BACK)).strftime("%Y-%m-%d")  # used if API supports date filtering directly


def fetch_page(params, timeout=60):
    """Fetch a single page from data.gov.in; returns DataFrame."""
    url = f"{API_BASE}?{urlencode(params, doseq=True)}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    if params.get("format", "json") == "csv":
        from io import StringIO
        return pd.read_csv(StringIO(resp.text))

    # Assume json structure with 'records'
    data = resp.json()
    records = data.get("records") or data.get("data") or []
    return pd.DataFrame(records)


def fetch_last_n_years(state=None, district=None, commodity=None, format_=FORMAT, limit=LIMIT):
    """
    Fetch last 7 years (once) with pagination. If the API supports date filters,
    we pass filters[Arrival_Date] (exact semantics depend on the resource).
    Otherwise we fetch all and filter locally by date.
    """
    filters = {}
    if state:
        filters["filters[State]"] = state
    if district:
        filters["filters[District]"] = district
    if commodity:
        filters["filters[Commodity]"] = commodity

    # Try server-side date filter if supported by your resource:
    # Some resources accept exact date or >=, but many don't.
    # If Arrival_Date supports equality only, we will filter locally afterward.
    # Uncomment if your resource supports range or min date:
    # filters["filters[Arrival_Date]"] = START_DATE

    # Base query
    base = {
        "api-key": API_KEY,
        "format": format_,
        "limit": limit,
        "offset": 0
    }
    base.update(filters)

    dfs = []
    page = 0
    while True:
        q = dict(base)
        q["offset"] = page * limit
        df_page = fetch_page(q)
        if df_page is None or df_page.empty:
            break
        dfs.append(df_page)
        page += 1
        # Sample key returns max 10 per call; stop when page smaller than limit
        if len(df_page) < limit:
            break
        time.sleep(SLEEP_SEC)

    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    # Normalize schema
    df = normalize_columns(df)

    # Filter locally to last 7 years
    cutoff = pd.Timestamp(START_DATE)
    df = df[df["Arrival_Date"] >= cutoff]

    return df


def normalize_columns(df):
    """
    Map incoming columns to the expected schema used in your trainer.
    Adjust keys if your resource uses different names (e.g. lowercase).
    """
    # Build case-insensitive lookup
    col_lookup = {c.lower(): c for c in df.columns}

    def pick(name):
        return col_lookup.get(name.lower())

    def get_col(name):
        src = pick(name)
        return df[src] if src in df.columns else pd.Series([np.nan] * len(df))

    out = pd.DataFrame({
        "State": get_col("State"),
        "District": get_col("District"),
        "Market": get_col("Market"),
        "Commodity": get_col("Commodity"),
        "Variety": get_col("Variety"),
        "Grade": get_col("Grade"),
        "Arrival_Date": get_col("Arrival_Date"),
        "Min_Price": pd.to_numeric(get_col("Min_Price"), errors="coerce"),
        "Max_Price": pd.to_numeric(get_col("Max_Price"), errors="coerce"),
        "Modal_Price": pd.to_numeric(get_col("Modal_Price"), errors="coerce"),
        "Commodity_Code": pd.to_numeric(get_col("Commodity_Code"), errors="coerce")
    })

    # Dates
    out["Arrival_Date"] = pd.to_datetime(out["Arrival_Date"], errors="coerce")
    out = out.dropna(subset=["Arrival_Date", "Modal_Price"])

    # Feature extraction
    out["Year"] = out["Arrival_Date"].dt.year
    out["Month"] = out["Arrival_Date"].dt.month
    out["Day"] = out["Arrival_Date"].dt.day

    return out


def train_and_save(df, model_name):
    print(f"Training on {len(df):,} rows")

    categorical_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    numeric_cols = ["Year", "Month", "Day", "Commodity_Code", "Min_Price", "Max_Price"]

    # Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[categorical_cols])
    X_num = df[numeric_cols].fillna(0).values
    X = np.hstack([X_cat, X_num])
    y = df["Modal_Price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelX = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    modelX.fit(X_train, y_train)

    # Evaluate
    y_pred = modelX.predict(X_test)
    mse2 = mean_squared_error(y_test, y_pred)
    rmse2 = np.sqrt(mse2)
    mae2 = mean_absolute_error(y_test, y_pred)
    r22 = r2_score(y_test, y_pred)

    print("\n=== XGBoost Performance Metrics ===")
    print(f"Mean Squared Error (MSE): {mse2:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse2:,.2f}")
    print(f"Mean Absolute Error (MAE): {mae2:,.2f}")
    print(f"R-squared (R²): {r22:.4f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolors="black", linewidth=0.5)
    plt.xlabel("Actual Price", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title("XGBoost: Actual vs Predicted Agricultural Prices", fontsize=14, fontweight="bold")
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction Line")
    plt.text(0.05, 0.95, f"R² = {r22:.4f}", transform=plt.gca().transAxes,
             fontsize=12, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(OUT_OUTPUTS_DIR, f"{model_name}_prediction.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save model bundle
    model_bundle = {
        "model": modelX,
        "encoder": encoder,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "performance": {"r2": float(r22), "rmse": float(rmse2), "mae": float(mae2)}
    }
    model_path = os.path.join(OUT_MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model_bundle, model_path)

    print(f"Saved model -> {model_path}")
    print(f"Saved plot  -> {plot_path}")
    return model_path, plot_path


def main():
    print("Fetching last 7 years from data.gov.in ...")
    df = fetch_last_n_years(state=STATE, district=DISTRICT, commodity=COMMODITY, format_=FORMAT, limit=LIMIT)
    if df.empty:
        print("No data returned. Check RESOURCE_ID, API key, and filters.")
        return

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Build a model name from scope
    parts = []
    if STATE: parts.append(f"state={STATE}")
    if DISTRICT: parts.append(f"district={DISTRICT}")
    if COMMODITY: parts.append(f"commodity={COMMODITY}")
    model_name = "xgb_api" if not parts else "xgb_api__" + "__".join(p.replace(" ", "_") for p in parts)

    train_and_save(df, model_name)


if __name__ == "__main__":
    main()
