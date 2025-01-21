import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model_uas.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Insurance Premium Predictor")

st.title("Insurance Premium Predictor")
st.write("NIM: 2019230099")
st.write("Nama: Muhammad Fahmi")

st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=27)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)

with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", options=["No", "Yes"])

sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

if st.button("Calculate Premium"):
    input_data = np.array([age, sex, bmi, children, smoker])
    input_data = input_data.reshape(1, -1)
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    
    st.subheader("Predicted Insurance Premium")
    st.write(f"${prediction[0]:,.2f} per month")
    
    st.write("---")
    st.write("Factors affecting the premium:")
    factors = []
    if age > 50:
        factors.append("- Higher age typically results in higher premiums")
    if bmi > 30:
        factors.append("- BMI above 30 (obese range) increases health risks")
    if smoker:
        factors.append("- Smoking significantly increases health risks and premiums")
    if children > 0:
        factors.append("- Number of children affects family coverage costs")
    
    if factors:
        for factor in factors:
            st.write(factor)