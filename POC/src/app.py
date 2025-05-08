import streamlit as st
from pickle import load
import numpy as np

with open("./models/Random_forest_model.sav", 'rb') as f:
    model = load(f)


st. title('Are you diabetic?')

Pregnancies = st. slider("Pregnancies", min_value = 0.0, max_value = 17.0, step = 1.0)
Glucose = st. slider ("Glucose", min_value = 0.0, max_value = 200.0, step = 1.0)
BloodPressure = st. slider("BloodPressure", min_value = 20.0, max_value = 120.0, step = 1.0)
# Insulin = st.slider ("Insulin", min_value = 0.0,max_value= 600.0,step= 10.0)
Insulin = st.text_input('Insulin level:')
Bmi = st. slider("Bmi", min_value = 10.0, max_value = 70.0, step = 0.5)
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree level:')
Age = st.text_input('Age:')

if st. button("Predict"):
    data_a_predecir = [[Pregnancies, Glucose, BloodPressure, Insulin, Bmi, DiabetesPedigreeFunction, Age]]
    prediction = model.predict(data_a_predecir)[0]
    if prediction == 0:
        prediction = 'NO'
    else:
        prediction = 'YES'
    st.write("Prediction", prediction)
