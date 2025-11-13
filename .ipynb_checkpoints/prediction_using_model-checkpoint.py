# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:03:42 2025

@author: Anirban Boral
"""

import joblib
import pandas as pd
import numpy as np

model1_data=joblib.load("Models/agri_modelX.joblib")

def predict_single_price(model_data, state, district, market, commodity, variety, grade, year, month, day, commodity_code, min_price, max_price):
    model=model_data["model"]
    encoder=model_data["encoder"]
    new_data = pd.DataFrame({
        'State': [state],
        'District': [district], 
        'Market': [market],
        'Commodity': [commodity],
        'Variety': [variety],
        'Grade': [grade],
        'Year': [year],
        'Month': [month], 
        'Day': [day],
        'Commodity_Code': [commodity_code],
        'Min_Price': [min_price],
        'Max_Price': [max_price]
    })
    
    categorical_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
    numeric_cols = ['Year', 'Month', 'Day', 'Commodity_Code', 'Min_Price', 'Max_Price']
    
    encoded_features = encoder.transform(new_data[categorical_cols])
    numeric_features = new_data[numeric_cols].values
    X_new = np.hstack([encoded_features, numeric_features])
    predicted_price = model.predict(X_new)[0]
    return predicted_price

predicted_priceX = predict_single_price(
    model_data=model1_data,
    state='Maharashtra',
    district='Nashik', 
    market='APMC Nashik',
    commodity='Onion',
    variety='Local',
    grade='Grade I',
    year=2025,
    month=10,
    day=25,
    commodity_code=100,
    min_price=1500,
    max_price=2500
)
print(f"Predicted XGBoost Price: {predicted_priceX:,.0f}")
