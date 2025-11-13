import streamlit as st
import pandas as pd
import os
import altair as alt
import joblib
import numpy as np
from datetime import date
from prediction_using_model import predict_modal_price

# ---------------------- STYLES ----------------------
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;900&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background: #0f1115;
            color: #e6eef8;
        }

        .main .block-container { padding-top: 14px; }

        .glass {
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            padding: 22px;
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: 0 4px 40px rgba(0,0,0,0.6);
        }

        .pred-box {
            background: rgba(255,255,255,0.04);
            border-left: 4px solid #38bdf8;
            padding: 20px;
            border-radius: 10px;
            margin-top: 18px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#0b0d10,#111318);
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        .stButton>button {
            background: linear-gradient(90deg,#0ea5a6,#38bdf8);
            color: #061220;
            font-weight: 700;
            border-radius: 10px;
            padding: 8px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------- DATA LOADER ----------------------
@st.cache_data
def load_data():
    folder = "Data"
    if not os.path.exists(folder):
        st.error("❌ 'Data' folder missing.")
        return pd.DataFrame()

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        st.error("❌ No CSV files found.")
        return pd.DataFrame()

    required = {"state","district","market","commodity","variety"}
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(folder,f))
            df.columns = df.columns.str.strip().str.lower()
            if required.issubset(df.columns):
                dfs.append(df)
        except:
            pass

    if not dfs:
        st.error("❌ CSVs missing required columns.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    for c in ["state","district","market","commodity","variety"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")

    for c in ["min_price","max_price","modal_price","commodity_code"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def safe_unique(x):
    return sorted(x.dropna().unique())

def infer_minmax(df):
    if df.empty:
        return None, None
    if "min_price" in df.columns and "max_price" in df.columns:
        r = df.dropna(subset=["min_price","max_price"])
        if not r.empty:
            row = r.iloc[-1]
            return row["min_price"], row["max_price"]
    if "modal_price" in df.columns:
        r = df.dropna(subset=["modal_price"])
        if not r.empty:
            mp = r.iloc[-1]["modal_price"]
            return mp*0.98, mp*1.02
    return None, None

# ---------------------- APP START ----------------------
st.set_page_config(page_title="Agri Predictor", layout="wide", page_icon="🌾")
load_css()

df = load_data()
if df.empty:
    st.stop()

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    page = st.radio(
        "🌿 Navigation",
        [" Price Prediction", " Market Trends", " Commodity Comparison"]
    )

# ---------------------- PAGE: PRICE PREDICTION ----------------------
if page == " Price Prediction":

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header(" Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox(" State", safe_unique(df["state"]))
        district = st.selectbox(" District", safe_unique(df[df["state"]==state]["district"]))

    with col2:
        market = st.selectbox(" Market",
                              safe_unique(df[(df["state"]==state)&(df["district"]==district)]["market"]))
        commodity = st.selectbox(" Commodity",
                                 safe_unique(df[(df["state"]==state)&(df["district"]==district)&(df["market"]==market)]["commodity"]))

    with col1:
        variety = st.selectbox(
            " Variety",
            safe_unique(df[(df["state"]==state)&(df["district"]==district)&
                           (df["market"]==market)&(df["commodity"]==commodity)]["variety"])
        )

    with col2:
        if "grade" in df.columns:
            grade = st.selectbox(" Grade", safe_unique(df["grade"]))
        else:
            grade = st.text_input(" Grade")

    picked = st.date_input(" Date", date.today())
    year, month, day = picked.year, picked.month, picked.day

    valid_rows = df[(df["state"]==state)&(df["district"]==district)&
                    (df["market"]==market)&(df["commodity"]==commodity)&
                    (df["variety"]==variety)]

    if "commodity_code" in df.columns:
        code_list = safe_unique(valid_rows["commodity_code"])
        commodity_code = st.selectbox(" Commodity Code (auto)", code_list if code_list else [0])
    else:
        commodity_code = st.number_input(" Commodity Code", value=0)

    auto_min, auto_max = infer_minmax(valid_rows)

    st.info(f"Suggested → Min: {auto_min}, Max: {auto_max}")

    manual_min = st.number_input("Enter Min Price", value=float(auto_min) if auto_min else 0.0)
    manual_max = st.number_input("Enter Max Price", value=float(auto_max) if auto_max else 0.0)

    if st.button(" Predict Price"):
        pred = predict_modal_price(
            state, district, market, commodity, variety, grade,
            year, month, day, commodity_code,
            min_price=manual_min,
            max_price=manual_max
        )

        pred = max(pred, 0)

        st.markdown(
            f"<div class='pred-box'><h2>💰 ₹{pred:.2f}</h2><p>Predicted modal price</p></div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PAGE: MARKET TRENDS ----------------------
elif page == " Market Trends":

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header(" Market Trends")

    state = st.selectbox("State", safe_unique(df["state"]))
    district = st.selectbox("District", safe_unique(df[df["state"]==state]["district"]))
    market = st.selectbox("Market", safe_unique(df[(df["state"]==state)&(df["district"]==district)]["market"]))

    sub = df[(df["state"]==state)&(df["district"]==district)&(df["market"]==market)]

    if not sub.empty:
        sub["arrival_date"] = pd.to_datetime(sub["arrival_date"], errors="coerce")
        chart = alt.Chart(sub).mark_line(point=True).encode(
            x="arrival_date:T",
            y="modal_price:Q",
            color="commodity:N",
            tooltip=["arrival_date","commodity","variety","modal_price"]
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PAGE: COMMODITY COMPARISON ----------------------
elif page == " Commodity Comparison":

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header(" Commodity Comparison")

    state = st.selectbox("State", safe_unique(df["state"]))
    district = st.selectbox("District", safe_unique(df[df["state"]==state]["district"]))
    market = st.selectbox("Market", safe_unique(df[(df["state"]==state)&(df["district"]==district)]["market"]))

    sub = df[(df["state"]==state)&(df["district"]==district)&(df["market"]==market)]

    if not sub.empty:
        comp = sub.groupby("commodity", as_index=False)["modal_price"].mean()
        chart = alt.Chart(comp).mark_bar().encode(
            x=alt.X("commodity:N", sort="-y"),
            y="modal_price:Q",
            tooltip=["commodity","modal_price"]
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
