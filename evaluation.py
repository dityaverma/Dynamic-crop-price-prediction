# evaluation.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_dataset_for_region(state, district):
    folder = "Data"
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    df_list = []
    for f in files:
        temp = pd.read_csv(os.path.join(folder, f))
        temp.columns = temp.columns.str.strip().str.lower()
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    # Lowercase value normalization
    df["state"] = df["state"].astype(str).str.lower().str.strip()
    df["district"] = df["district"].astype(str).str.lower().str.strip()

    # Filter region
    df = df[(df["state"] == state.lower()) & (df["district"] == district.lower())]

    return df


def evaluate_model(state, district):
    tag = f"ProjectData{district.replace(' ', '')}{state.replace(' ', '')}"
    model_path = os.path.join("Models", f"{tag}.joblib")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    bundle = joblib.load(model_path)

    model = bundle["model"]
    enc = bundle["encoder"]
    categorical_cols = [c.lower() for c in bundle["categorical_cols"]]
    numeric_cols = [c.lower() for c in bundle["numeric_cols"]]

    df = load_dataset_for_region(state, district)
    if df.empty:
        print("No data in dataset for this region.")
        return

    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("Unknown")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X_cat = enc.transform(df[categorical_cols])
    X_num = df[numeric_cols].values
    X = np.hstack([X_cat, X_num])

    y = df["modal_price"].values
    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print("\n--- Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")


if __name__ == "__main__":
    evaluate_model("Maharashtra", "Nashik")
