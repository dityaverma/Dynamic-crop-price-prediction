# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:03:42 2025

@author: Anirban Boral
"""

import joblib
import pandas as pd
import numpy as np
import os
def load_model_for_region(state: str, district: str):
    """
    Returns the appropriate trained model bundle (.joblib).
    Routes Sikkim to a proxy (Kolkata, West Bengal) due to low data.
    """
    state = (state or "").replace(" ", "")
    district = (district or "").replace(" ", "")

    if state.lower() == "sikkim":
        return joblib.load("Models/ProjectDataKolkataWestBengal.joblib")

    tag = f"ProjectData{district}{state}"
    path = os.path.join("Models", f"{tag}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def predict_modal_price(state, district, market, commodity, variety, grade,
                        year, month, day, commodity_code, min_price=None, max_price=None):
    bundle = load_model_for_region(state, district)
    model = bundle["model"]
    enc = bundle["encoder"]
    categorical_cols = bundle["categorical_cols"]
    numeric_cols = bundle["numeric_cols"]

    row = {
        "State": state, "District": district, "Market": market,
        "Commodity": commodity, "Variety": variety, "Grade": grade,
        "Year": year, "Month": month, "Day": day,
        "Commodity_Code": commodity_code
    }
    if "Min_Price" in numeric_cols:
        row["Min_Price"] = min_price or 0
    if "Max_Price" in numeric_cols:
        row["Max_Price"] = max_price or 0

    df = pd.DataFrame([row])
    X_cat = enc.transform(df[categorical_cols])
    X_num = df[numeric_cols].fillna(0).values
    X = np.hstack([X_cat, X_num])

    pred = model.predict(X)[0]
    return float(pred)

if __name__ == "__main__":
    price = predict_modal_price(
        state="Sikkim", district="East Sikkim", market="Gangtok",
        commodity="Onion", variety="Local", grade="Grade I",
        year=2025, month=11, day=10, commodity_code=100,
        min_price=1500, max_price=2600  
    )
    print(f"Predicted Modal Price: ₹{price:,.0f}")
