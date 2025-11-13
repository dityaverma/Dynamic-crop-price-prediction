# dynamic-agriculture-price-prediction

## Project Description
This project predicts the modal price of agricultural commodities using machine learning. It leverages historical market data, commodity details, and arrival dates. 

## Technologies Used
- Python, scikit-learn, XGBoost
- Pandas, NumPy for data processing
- Matplotlib for visualization
- Joblib for model persistence

## Dataset
- The dataset used here is from Nashik district, Maharashtra, collected from data.gov.in.
- Provided data includes features such as `Market`, `Commodity`, `Variety`, `Grade`, `Arrival_Date`, and `Commodity_Code`.
- Target variable is `Modal_Price`.
- Data preprocessing includes encoding of categorical variables and date transformation.

## Features
- Complete data preprocessing pipeline
- ML models (Linear Regression, XGBoost)
- Model persistence with joblib
- Production-ready code structure

## Installation and Usage
- Open prediction_using_model.py
- Run it
