"""
Created on Sat Aug 30 19:09:49 2025

@author: Anirban Boral
"""


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib

# ---------------- CONFIG ----------------
DATA_DIR = "Data"
MODELS_DIR = "Models"
OUTPUTS_DIR = "Outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

USE_MIN_MAX = True 

LR_FALLBACK_THRESHOLD = 3000

MIN_ROWS = 500

# -------------- HELPERS -----------------
def load_df(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_excel(p)

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Map columns case-insensitively
    cols = {c.lower(): c for c in df.columns}
    def pick(n): return cols.get(n.lower())
    def col(n):
        c = pick(n)
        return df[c] if c in df.columns else np.nan

    out = pd.DataFrame({
        "State": col("State"),
        "District": col("District"),
        "Market": col("Market"),
        "Commodity": col("Commodity"),
        "Variety": col("Variety"),
        "Grade": col("Grade"),
        # Many sheets are dd/mm/yyyy -> set dayfirst=True to avoid warnings
        "Arrival_Date": pd.to_datetime(col("Arrival_Date"), errors="coerce", dayfirst=True),
        "Modal_Price": pd.to_numeric(col("Modal_Price"), errors="coerce"),
        "Commodity_Code": pd.to_numeric(col("Commodity_Code"), errors="coerce"),
    })
    if USE_MIN_MAX:
        out["Min_Price"] = pd.to_numeric(col("Min_Price"), errors="coerce")
        out["Max_Price"] = pd.to_numeric(col("Max_Price"), errors="coerce")

    # Keep rows with valid date and target
    out = out.dropna(subset=["Arrival_Date", "Modal_Price"])

    # Date features
    out["Year"]  = out["Arrival_Date"].dt.year
    out["Month"] = out["Arrival_Date"].dt.month
    out["Day"]   = out["Arrival_Date"].dt.day
    return out

def train_and_save(df: pd.DataFrame, tag: str):
    categorical = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    numeric = ["Year", "Month", "Day", "Commodity_Code"]
    if USE_MIN_MAX:
        numeric += ["Min_Price", "Max_Price"]

    # Encoding
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df[categorical])
    X_num = df[numeric].fillna(0).values
    X = np.hstack([X_cat, X_num])
    y = df["Modal_Price"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose algorithm
    use_lr = len(df) < LR_FALLBACK_THRESHOLD
    if use_lr:
        model = LinearRegression()
        algo = "LR"
    else:
        model = XGBRegressor(
            n_estimators=300 if len(df) < 100000 else 500,
            max_depth=6 if len(df) < 100000 else 7,
            learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist",
            random_state=42, n_jobs=-1
        )
        algo = "XGB"

    model.fit(X_tr, y_tr)

    # Evaluate
    y_pr = model.predict(X_te)
    mse = mean_squared_error(y_te, y_pr)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_te, y_pr))
    r2 = float(r2_score(y_te, y_pr))
    print(f"{tag}: [{algo}] R2={r2:.4f} RMSE={rmse:.2f} MAE={mae:.2f} rows={len(df)}")

    # Plot
    plt.figure(figsize=(8,7))
    plt.scatter(y_te, y_pr, s=6, alpha=0.6, edgecolors="k", linewidth=0.2)
    mn, mx = float(min(y_te.min(), y_pr.min())), float(max(y_te.max(), y_pr.max()))
    plt.plot([mn, mx], [mn, mx], "r--", lw=2)
    plt.title(f"Actual vs Predicted ({tag}) [{algo}]")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.grid(True, alpha=0.3)
    plt.text(0.04, 0.96, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
             fontsize=11, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{tag}_prediction.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save model bundle
    bundle = {
        "model": model,
        "encoder": enc,
        "categorical_cols": categorical,
        "numeric_cols": numeric,
        "performance": {"r2": r2, "rmse": rmse, "mae": mae},
        "algo": algo
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, f"{tag}.joblib"))

def main():
    files = [p for p in Path(DATA_DIR).iterdir()
             if p.is_file() and p.stem.startswith("ProjectData") and p.suffix.lower() in (".csv",".xlsx",".xls")]
    if not files:
        print("No ProjectData*.csv/.xlsx files found in Data/")
        return

    summary = []
    for p in sorted(files):
        try:
            df = load_df(p)
            df = prepare(df)
            if len(df) < MIN_ROWS:
                print(f"Skip {p.name} (only {len(df)} usable rows)")
                continue
            tag = p.stem
            
            if "Sikkim" in tag and len(df) < 500:
                print("Using KolkataWestBengal model as proxy for Sikkim due to low data")
            
            train_and_save(df, tag)
            summary.append({"file": p.name, "rows": len(df)})
        except Exception as e:
            print(f"Failed on {p.name}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(OUTPUTS_DIR, "training_summary.csv"), index=False)
        print("Saved training_summary.csv")

if __name__ == "__main__":
    main()
