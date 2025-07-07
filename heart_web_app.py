import pickle
import numpy as np
import streamlit as st

# Load the model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def heart_disease_prediction(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return 'The person has Heart Disease' if prediction[0] == 1 else 'The person does not have Heart Disease'

# Streamlit UI
st.title("Heart Disease Prediction App")

# Input fields
age = st.number_input('Age')
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cp = st.number_input('Chest Pain Type (0–3)', 0, 3)
trestbps = st.number_input('Resting Blood Pressure')
chol = st.number_input('Serum Cholesterol (mg/dl)')
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)', [0, 1])
restecg = st.selectbox('Resting ECG (0–2)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved')
exang = st.selectbox('Exercise Induced Angina (1 = Yes; 0 = No)', [0, 1])
oldpeak = st.number_input('ST Depression')
slope = st.selectbox('Slope (0–2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0–3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', [1, 2, 3])

# Button
if st.button('Predict'):
    result = heart_disease_prediction((age, sex, cp, trestbps, chol, fbs, restecg,
                                       thalach, exang, oldpeak, slope, ca, thal))
    st.success(result)
