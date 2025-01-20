import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# App Title
st.title("Osteoporosis Risk Prediction")
st.write("This application predicts the risk of osteoporosis based on the input features provided.")

# User Input Section
st.header("Enter Patient Details")

# User Input for Features
feature_inputs = {}
feature_inputs['Age'] = st.number_input("Age", min_value=0, max_value=120, value=30)
feature_inputs['Gender'] = st.selectbox("Gender", ["Male", "Female"])
feature_inputs['Hormonal Changes'] = st.selectbox("Hormonal Changes", ["Normal", "Postmenopausal"])
feature_inputs['Family History'] = st.selectbox("Family History of Osteoporosis", ["No", "Yes"])
feature_inputs['Race/Ethnicity'] = st.selectbox("Race/Ethnicity", ["Asian", "Caucasian", "African American"])
feature_inputs['Body Weight'] = st.selectbox("Body Weight", ["Underweight", "Normal"])
feature_inputs['Calcium Intake'] = st.selectbox("Calcium Intake", ["Low", "Adequate"])
feature_inputs['Vitamin D Intake'] = st.selectbox("Vitamin D Intake", ["Insufficient", "Sufficient"])
feature_inputs['Physical Activity'] = st.selectbox("Physical Activity", ["Sedentary", "Active"])
feature_inputs['Smoking'] = st.selectbox("Smoking Status", ["No", "Yes"])
feature_inputs['Alcohol Consumption'] = st.selectbox("Alcohol Consumption", ["None", "Moderate"])
feature_inputs['Medical Conditions'] = st.selectbox(
    "Medical Conditions", ["None", "Hyperthyroidism", "Rheumatoid Arthritis"])
feature_inputs['Medications'] = st.selectbox("Medications", ["None", "Corticosteroids"])
feature_inputs['Prior Fractures'] = st.selectbox("Prior Fractures", ["No", "Yes"])

# Encoding Categorical Features
feature_map = {
    'Gender': {'Male': 0, 'Female': 1},
    'Hormonal Changes': {'Normal': 0, 'Postmenopausal': 1},
    'Family History': {'No': 0, 'Yes': 1},
    'Race/Ethnicity': {'Asian': 0, 'Caucasian': 1, 'African American': 2},
    'Body Weight': {'Underweight': 0, 'Normal': 1},
    'Calcium Intake': {'Low': 0, 'Adequate': 1},
    'Vitamin D Intake': {'Insufficient': 0, 'Sufficient': 1},
    'Physical Activity': {'Sedentary': 0, 'Active': 1},
    'Smoking': {'No': 0, 'Yes': 1},
    'Alcohol Consumption': {'None': 0, 'Moderate': 1},
    'Medical Conditions': {'None': 0, 'Hyperthyroidism': 1, 'Rheumatoid Arthritis': 2},
    'Medications': {'None': 0, 'Corticosteroids': 1},
    'Prior Fractures': {'No': 0, 'Yes': 1}
}

for key, mapping in feature_map.items():
    feature_inputs[key] = mapping[feature_inputs[key]]

# Organizing inputs into a DataFrame
input_data = pd.DataFrame([feature_inputs])

# Scaling the input data
scaled_data = scaler.transform(input_data)

# Prediction
if st.button("Predict Osteoporosis Risk"):
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[:, 1]

    if prediction[0] == 1:
        st.error(f"High Risk of Osteoporosis (Probability: {probability[0]*100:.2f}%)")
    else:
        st.success(f"Low Risk of Osteoporosis (Probability: {probability[0]*100:.2f}%)")