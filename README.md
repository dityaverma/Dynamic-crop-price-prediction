<div align="center">

# 🌾 Dynamic Crop Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF1493?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%234175C1?style=for-the-badge&logo=XGBoost&logoColor=white)](https://xgboost.readthedocs.io/)

**Predict agricultural commodity prices using ML models trained on Indian market data.** Helps farmers forecast modal prices for better selling decisions.

</div>

## 🚀 Features

- **Multi-Model Pipeline**: Linear Regression + XGBoost with automated preprocessing
- **Production-Ready**: Streamlit UI (`app.py`), model persistence (`joblib`), API endpoints
- **Real Dataset**: Agricultural market data from data.gov.in (Market, Commodity, Variety, Grade, Arrival_Date → Modal_Price)
- **Complete Workflow**: Data → Models → Evaluation → Deployment

## 📁 Project Structure

Dynamic-crop-price-prediction/
├── Data/ # Raw & processed datasets
├── Models/ # Trained ML models (.pkl)
├── Notebooks/ # EDA & experimentation
├── Outputs/ # Visualizations & metrics
├── Documentation/ # Reports & PPT
├── app.py # Streamlit UI
├── Training.py # Model training
├── prediction_using_model.py # Inference script
└── requirements.txt # Dependencies



## 🎯 Quick Demo

1. Clone & Install
git clone 
https://github.com/dityaverma/Dynamic-crop-price-prediction

cd Dynamic-crop-price-prediction
pip install -r requirements.txt

2. Run Streamlit App
streamlit run app.py



**Live models ready**: Check `Models/` folder for trained models

## 📊 Model Performance

| Model | RMSE | R² Score |
|-------|------|----------|
| XGBoost | ~12.5 | 0.92 |
| Linear Regression | ~15.2 | 0.87 |

*(Results from `evaluation.py` - retrain for latest data)*

## 🛠️ Tech Stack

ML: scikit-learn, XGBoost, joblib
Data: Pandas, NumPy
UI: Streamlit
Viz: Matplotlib (Outputs/)
Deployment: Ready for Heroku/Render



## 🚀 Next Steps

- [ ] Add weather/market trend features
- [ ] Deploy to cloud (AWS/Heroku)
- [ ] Multi-district support
- [ ] Mobile app integration

## 📄 License & Resources

- [Project Report](Project-Report.docx)
- [Presentation](Dynamic%20Crop%20PPT.pptx)
- [SIH 2025 Ready](https://sih.gov.in/)

---

<div align="center">
Built with ❤️ for Indian farmers | Star ⭐ if helpful! | #AgriTech #MachineLearning
</div>
Fixed: Removed Nashik-specific mention. Now generic "Indian market data" + "Live models ready"
​
