import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="CAFV Eligibility Predictor",
    page_icon="🚗",
    layout="centered"
)

# ------------------ CUSTOM STYLING ------------------
page_style = """
<style>
/* Whole page background */
[data-testid="stAppViewContainer"] {
    background-color: #f5f5f9;
}

/* Title styling */
.title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #2b2b2b;
    text-align: center;
    margin-bottom: 15px;
}

/* Description text */
.subtitle {
    font-size: 16px !important;
    color: #555;
    text-align: center;
    margin-bottom: 30px;
}

/* Input area box styling */
.box {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Prediction box */
.result-box {
    background-color: #eafbea;
    padding: 20px;
    border-radius: 16px;
    border-left: 6px solid #3ac47d;
    font-size: 18px;
    font-weight: 600;
    color: #1f6b43;
}

/* Meaning box */
.meaning-box {
    background-color: #eef3ff;
    padding: 18px;
    border-radius: 16px;
    border-left: 6px solid #4a6ee0;
    color: #2e4bb3;
    font-size: 16px;
    font-weight: 500;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1 class='title'>🚗 CAFV Eligibility Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter electric vehicle details to predict CAFV eligibility</p>", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = joblib.load("rf_best.pkl")

# ---- encoding maps (as you provided) ----
county_map = {"King": 1}

city_map = {
    "Bellevue": 0, "Kirkland": 1, "Redmond": 2,
    "Sammamish": 3, "Seattle": 4
}

make_map = {
    "BMW":0,"CHEVROLET":1,"FORD":2,"HYUNDAI":3,"KIA":4,
    "NISSAN":5,"RIVIAN":6,"TESLA":7,"TOYOTA":8
}

model_map = {
    "BOLT EV":0,"IONIQ 5":1,"LEAF":2,"MODEL 3":3,"MODEL S":4,
    "MODEL X":5,"MODEL Y":6,"MUSTANG MACH-E":7,"NIRO":8,
    "R1S":9,"RAV4 PRIME (PHEV)":10,"X5":11
}

ev_type_map = {
    "Battery Electric Vehicle (BEV)": 0,
    "Plug-in Hybrid Electric Vehicle (PHEV)": 1
}

postal_map = {
    "98004":0,"98006":1,"98033":2,"98034":3,"98052":4,
    "98074":5,"98075":6,"98125":7
}

utility_map = {
    "CITY OF SEATTLE - (WA) | CITY OF TACOMA - (WA)":0,
    "PUGET SOUND ENERGY INC | CITY OF TACOMA - (WA)":1
}

cafv_map = {
    0: "Clean Alternative Fuel Vehicle Eligible",
    1: "Eligibility unknown as battery range has not been researched"
}

feature_order = ['County', 'City', 'State', 'Postal Code', 'Make', 'Model',
                 'Electric Vehicle Type', 'Electric Range', 'Electric Utility', 'Age']

# ------------------ INPUT FORM ------------------
st.markdown("<div class='box'>", unsafe_allow_html=True)

county = st.selectbox("County", list(county_map.keys()))
city = st.selectbox("City", list(city_map.keys()))
state = st.selectbox("State", ["WA"])
postal = st.selectbox("Postal Code", list(postal_map.keys()))

make = st.selectbox("Make", list(make_map.keys()))
model_name = st.selectbox("Model", list(model_map.keys()))

ev_type = st.selectbox("Electric Vehicle Type", list(ev_type_map.keys()))
utility = st.selectbox("Electric Utility", list(utility_map.keys()))

electric_range = st.number_input("Electric Range", min_value=0)
age = st.number_input("Vehicle Age", min_value=0)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if st.button("Predict CAFV Eligibility"):
    
    input_row = [
        county_map[county],
        city_map[city],
        0,                          # WA fixed
        postal_map[postal],
        make_map[make],
        model_map[model_name],
        ev_type_map[ev_type],
        electric_range,
        utility_map[utility],
        age
    ]
    
    final_df = pd.DataFrame([input_row], columns=feature_order)
    
    pred = model.predict(final_df)[0]
    meaning = cafv_map[pred]
    prob = model.predict_proba(final_df)[0][pred]

    st.markdown(f"<div class='result-box'>Prediction: {pred} &nbsp;&nbsp;•&nbsp;&nbsp; Confidence: {prob:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='meaning-box'>Meaning: {meaning}</div>", unsafe_allow_html=True)
