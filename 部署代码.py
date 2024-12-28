# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:01:20 2024

@author: 86185
"""


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('CatBoost.pkl')

# Define feature options
Age_options = {
    1: '≤35 (1)',
    2: '＞35 (2)',
}
Occupation_options = {
    1: '有稳定工作 (1)',
    2: '无稳定工作 (2)'
}
Method_of_delivery_options = {
    1: '顺产 (1)',
    2: '剖宫产 (2)',
}
Marital_status_options = {
    1: '已婚 (1)',
    2: '未婚 (2)',
}
Educational_degree_options = {
    1: '专科及以下 (1)',
    2: '本科及以上 (2)',
}
average_monthly_household_income_options = {
    1: '≤5000 (1)',
    2: '＞5000 (2)',
}
medical_insurance_options = {
    1: 'No (1)',
    2: 'Yes (2)',
}
mode_of_conception_options = {
    1: '自然受孕 (1)',
    2: '人工受孕 (2)',
}
Pregnancy_complications_options = {
    1: 'Yes',
    2: 'No'
}
Breastfeeding_options = {
    1: 'Yes',
    2: 'No'
}
rooming_in_options = {
    1: 'Yes',
    2: 'No'
}
Planned_pregnancy_options = {
    1: 'Yes',
    2: 'No'
}

# Define feature names
feature_names = [
    "Intrapartum pain", "Postpartum pain", "Resilience", "Family support", "Psychological birth trauma","Age","Occupation","Method of delivery","Marital status","Educational degree","Average monthly household income","Medical insurance","Mode of conception","Pregnancy complications","Breastfeeding","Rooming-in","Planned pregnancy",
]

# Streamlit user interface
st.title("Negative Emotions Predictor")

# Intrapartum pain: numerical input
Intrapartum_pain = st.number_input("Intrapartum pain:", min_value=0, max_value=10, value=5)

# Postpartum pain: numerical input
Postpartum_pain = st.number_input("Postpartum pain:", min_value=0, max_value=10, value=5)

# Resilience: numerical input
Resilience = st.number_input("Resilience:", min_value=6, max_value=30, value=18)

# Family support: numerical input
Family_support = st.number_input("Family support:", min_value=0, max_value=10, value=5)

# Psychological birth trauma: numerical input
Psychological_birth_trauma = st.number_input("Psychological birth trauma:", min_value=0, max_value=42, value=14)

# Age: categorical selection
Age = st.selectbox("Age (1=≤35, 2=＞35):", options=[1, 2], format_func=lambda x: '≤35 (1)' if x == 1 else '＞35 (2)')

# Occupation: categorical selection
Occupation = st.selectbox("Occupation (1=有稳定工作, 2=无稳定工作):", options=[1, 2], format_func=lambda x: '有稳定工作 (1)' if x == 1 else '无稳定工作 (2)')

# Method of delivery: categorical selection
Method_of_delivery = st.selectbox("Method of delivery (1=顺产, 2=剖宫产):", options=[1, 2], format_func=lambda x: '顺产 (1)' if x == 1 else '剖宫产 (2)')

# Marital status: categorical selection
Marital_status = st.selectbox("Marital status (1=已婚, 2=未婚):", options=[1, 2], format_func=lambda x: '已婚 (1)' if x == 1 else '未婚 (2)')

# Educational degree: categorical selection
Educational_degree = st.selectbox("Educational degree (1=专科及以下, 2=本科及以上):", options=[1, 2], format_func=lambda x: '专科及以下 (1)' if x == 1 else '本科及以上 (2)')

# Average monthly household income: categorical selection
Average_monthly_household_income = st.selectbox("Average monthly household income (1=≤5000, 2=＞5000):", options=[1, 2], format_func=lambda x: '≤5000 (1)' if x == 1 else '＞5000 (2)')

# Medical insurance: categorical selection
Medical_insurance = st.selectbox("Medical insurance (1=No, 2=Yes):", options=[1, 2], format_func=lambda x: 'No (1)' if x == 1 else 'Yes (2)')

# Mode of conception: categorical selection
Mode_of_conception = st.selectbox("Mode of conception (1=自然受孕, 2=人工受孕):", options=[1, 2], format_func=lambda x: '自然受孕 (1)' if x == 1 else '人工受孕 (2)')

# Pregnancy complications: categorical selection
Pregnancy_complications = st.selectbox("Pregnancy complications (1=Yes, 2=No):", options=[1, 2], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (2)')

# Breastfeeding: categorical selection
Breastfeeding = st.selectbox("Breastfeeding (1=Yes, 2=No):", options=[1, 2], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (2)')

# Rooming-in: categorical selection
Rooming_in = st.selectbox("Rooming-in (1=Yes, 2=No):", options=[1, 2], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (2)')

# Process inputs and make predictions
feature_values = [Intrapartum_pain, Postpartum_pain, Resilience, Family_support, Psychological_birth_trauma, Age, Occupation, Method_of_delivery, Marital_status, Educational_degree, Average_monthly_household_income, Medical_insurance, Mode_of_conception, Pregnancy_complications, Breastfeeding, Rooming_in]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
    

