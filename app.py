import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Parkinson's Disease Telemonitoring", layout="wide")
st.title("Dự đoán UPDRS cho bệnh nhân Parkinson")
st.write("Nhập các chỉ số giọng nói để dự đoán motor_UPDRS và total_UPDRS.")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('models/scaler.pkl')
    model_motor = joblib.load('models/motor_updrs_model_RandomForest.pkl')
    model_total = joblib.load('models/total_updrs_model_RandomForest.pkl')
    return scaler, model_motor, model_total

try:
    scaler, model_motor, model_total = load_artifacts()
except Exception as e:
    st.error(f"Lỗi khi load model: {e}. Vui lòng kiểm tra lại thư mục 'models/'.")
    st.stop()

st.header("Thông tin đầu vào")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Thông tin chung")
    age = st.number_input("Age", min_value=0, max_value=120, value=55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    test_time = st.number_input("Test Time (days)", min_value=0.0, value=81.731)
    RPDE = st.number_input("RPDE", format="%.6f", value=0.59148)
    DFA = st.number_input("DFA", format="%.6f", value=0.73197)
    PPE = st.number_input("PPE", format="%.6f", value=0.310560)

with col2:
    st.subheader("Jitter")
    jitter_percent = st.number_input("Jitter(%)", format="%.6f", value=0.007395)
    jitter_abs = st.number_input("Jitter(Abs)", format="%.6f", value=0.000068)
    jitter_rap = st.number_input("Jitter:RAP", format="%.5f", value=0.007395)
    jitter_ppq5 = st.number_input("Jitter:PPQ5", format="%.5f", value=0.007395)
    jitter_ddp = st.number_input("Jitter:DDP", format="%.5f", value=0.007395)

with col3:
    st.subheader("Shimmer & Voice")
    shimmer = st.number_input("Shimmer", format="%.6f", value=0.028110)
    shimmer_db = st.number_input("Shimmer(dB)", format="%.6f", value=0.27)
    shimmer_apq3 = st.number_input("Shimmer:APQ3", format="%.6f", value=0.028110)
    shimmer_apq5 = st.number_input("Shimmer:APQ5", format="%.6f", value=0.028110)
    shimmer_apq11 = st.number_input("Shimmer:APQ11", format="%.6f", value=0.028110)
    shimmer_dda = st.number_input("Shimmer:DDA", format="%.6f", value=0.028110)
    nhr = st.number_input("NHR", format="%.6f", value=0.015734)
    hnr = st.number_input("HNR", format="%.6f", value=20.326)

def feature_engineering(raw_df):
    df = raw_df.copy()
    
    df['Jitter(avg)'] = df[['Jitter(%)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP']].mean(axis=1)
    df['Shimmer(avg)'] = df[['Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA']].mean(axis=1)
    
    expected_cols = [
        'age', 'sex', 'test_time', 'Jitter(Abs)', 'Shimmer(dB)', 'NHR', 'HNR', 
        'RPDE', 'DFA', 'PPE', 'Jitter(avg)', 'Shimmer(avg)'
    ]
    
    return df[expected_cols]

if st.button("Dự đoán UPDRS", type="primary", use_container_width=True):
    
    input_data = pd.DataFrame([{
        'age': age, 'test_time': test_time,
        'Jitter(%)': jitter_percent, 'Jitter(Abs)': jitter_abs, 'Jitter:RAP': jitter_rap, 'Jitter:PPQ5': jitter_ppq5, 'Jitter:DDP': jitter_ddp,
        'Shimmer': shimmer, 'Shimmer(dB)': shimmer_db, 'Shimmer:APQ3': shimmer_apq3, 'Shimmer:APQ5': shimmer_apq5, 'Shimmer:APQ11': shimmer_apq11, 'Shimmer:DDA': shimmer_dda,
        'NHR': nhr, 'sex': sex, 'HNR': hnr, 'RPDE': RPDE, 'DFA': DFA, 'PPE': PPE
    }])
    
    processed_df = feature_engineering(input_data)
    scaled_data = scaler.transform(processed_df)
    
    pred_motor = model_motor.predict(scaled_data)
    pred_total = model_total.predict(scaled_data)
    
    st.divider()
    st.subheader("Kết quả dự đoán")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Motor UPDRS Score", value=f"{pred_motor[0]:.2f}")
    with res_col2:
        st.metric(label="Total UPDRS Score", value=f"{pred_total[0]:.2f}")
