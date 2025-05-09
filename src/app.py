import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(project_root, 'models', 'xgboost_final.joblib')
    scaler_path = os.path.join(project_root, 'data', 'processed', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Get feature names from the model's booster
    if hasattr(model, 'get_booster'):
        feature_names = model.get_booster().feature_names
    else:
        # Fall back to saved feature names file
        features_path = os.path.join(project_root, 'models', 'exact_features.txt')
        with open(features_path, 'r') as f:
            feature_names = [line.strip() for line in f]
            
    return model, scaler, feature_names

model, scaler, expected_features = load_model_and_scaler()

st.title("Bank Term Deposit Subscription Predictor")
st.write("Enter customer information to predict if they will subscribe to a term deposit")

# Create form with two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.slider("Age", min_value=18, max_value=95, value=40)
    job = st.selectbox("Job", options=[
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown'
    ])
    marital = st.selectbox("Marital Status", options=['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox("Education Level", options=['primary', 'secondary', 'tertiary', 'unknown'])
    
    # Housing and loan
    housing = st.selectbox("Has housing loan?", options=['yes', 'no', 'unknown'])
    loan = st.selectbox("Has personal loan?", options=['yes', 'no', 'unknown'])

with col2:
    st.subheader("Campaign Information")
    contact = st.selectbox("Contact Type", options=['cellular', 'telephone'])
    month = st.selectbox("Month of Contact", options=[
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    duration = st.slider("Call Duration (seconds)", min_value=0, max_value=3000, value=180)
    campaign = st.slider("Number of Contacts", min_value=1, max_value=50, value=2)
    previous = st.slider("Previous Contacts", min_value=0, max_value=7, value=0)
    poutcome = st.selectbox("Previous Outcome", options=['failure', 'nonexistent', 'success'])
    
    # Economic indicators
    cons_price_idx = st.slider("Consumer Price Index", min_value=92.0, max_value=95.0, value=93.5, step=0.1)
    cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.0, max_value=-30.0, value=-40.0, step=0.5)
    nr_employed = st.slider("Number of Employees (thousands)", min_value=4900, max_value=5300, value=5100, step=10)

# Prediction button
if st.button("Predict Subscription"):
    try:
        # Create a DataFrame with all expected features initialized to 0
        input_data = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Fill in numerical features
        numerical_features = ['age', 'duration', 'campaign', 'previous', 
                            'cons.price.idx', 'cons.conf.idx', 'nr.employed']
        
        input_data['age'] = age
        input_data['duration'] = duration
        input_data['campaign'] = campaign
        input_data['previous'] = previous
        input_data['cons.price.idx'] = cons_price_idx
        input_data['cons.conf.idx'] = cons_conf_idx
        input_data['nr.employed'] = nr_employed
        
        # Binary features
        input_data['had_previous_contact'] = 1 if previous > 0 else 0
        
        # Handle all possible one-hot encoded features
        # Month
        for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
            if f'month_{m}' in expected_features:
                input_data[f'month_{m}'] = 1 if m == month else 0
                
        # Job
        for j in ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                 'retired', 'self-employed', 'services', 'student', 'technician',
                 'unemployed', 'unknown']:
            if f'job_{j}' in expected_features:
                input_data[f'job_{j}'] = 1 if j == job else 0
                
        # Marital
        for m in ['divorced', 'married', 'single', 'unknown']:
            if f'marital_{m}' in expected_features:
                input_data[f'marital_{m}'] = 1 if m == marital else 0
                
        # Education
        for e in ['primary', 'secondary', 'tertiary', 'unknown']:
            if f'education_{e}' in expected_features:
                input_data[f'education_{e}'] = 1 if e == education else 0
                
        # Housing
        for h in ['yes', 'no', 'unknown']:
            if f'housing_{h}' in expected_features:
                input_data[f'housing_{h}'] = 1 if h == housing else 0
                
        # Loan
        for l in ['yes', 'no', 'unknown']:
            if f'loan_{l}' in expected_features:
                input_data[f'loan_{l}'] = 1 if l == loan else 0
                
        # Contact
        for c in ['cellular', 'telephone']:
            if f'contact_{c}' in expected_features:
                input_data[f'contact_{c}'] = 1 if c == contact else 0
                
        # Poutcome
        for p in ['failure', 'nonexistent', 'success']:
            if f'poutcome_{p}' in expected_features:
                input_data[f'poutcome_{p}'] = 1 if p == poutcome else 0
        
        # Duration category
        if duration <= 120:
            duration_cat = 'short'
        elif duration <= 300:
            duration_cat = 'medium'
        else:
            duration_cat = 'long'
            
        for dc in ['short', 'medium', 'long']:
            if f'duration_category_{dc}' in expected_features:
                input_data[f'duration_category_{dc}'] = 1 if dc == duration_cat else 0
        
        # Age group
        if age <= 25:
            age_group = 'young'
        elif age <= 35:
            age_group = 'young_adult'
        elif age <= 50:
            age_group = 'middle_aged'
        else:
            age_group = 'senior'
            
        for ag in ['young', 'young_adult', 'middle_aged', 'senior']:
            if f'age_group_{ag}' in expected_features:
                input_data[f'age_group_{ag}'] = 1 if ag == age_group else 0
                
        # Campaign intensity
        if campaign <= 2:
            intensity = 'low'
        elif campaign <= 5:
            intensity = 'medium'
        else:
            intensity = 'high'
            
        for ci in ['low', 'medium', 'high']:
            if f'campaign_intensity_{ci}' in expected_features:
                input_data[f'campaign_intensity_{ci}'] = 1 if ci == intensity else 0
        
        # Season from month
        season_map = {
            'mar': 'spring', 'apr': 'spring', 'may': 'spring',
            'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
            'sep': 'autumn', 'oct': 'autumn', 'nov': 'autumn',
            'dec': 'winter', 'jan': 'winter', 'feb': 'winter'
        }
        season = season_map[month]
        
        for s in ['spring', 'summer', 'autumn', 'winter']:
            if f'season_{s}' in expected_features:
                input_data[f'season_{s}'] = 1 if s == season else 0
        
        # Scale numerical features
        numerical_cols_to_scale = [col for col in numerical_features if col in input_data.columns]
        input_data[numerical_cols_to_scale] = scaler.transform(input_data[numerical_cols_to_scale])
        
        # Verify we have all expected features
        missing_features = set(expected_features) - set(input_data.columns)
        extra_features = set(input_data.columns) - set(expected_features)
        
        if missing_features:
            st.error(f"Missing features: {missing_features}")
        if extra_features:
            st.error(f"Extra features: {extra_features}")
        
        # Verify features match exactly
        if set(input_data.columns) != set(expected_features):
            missing = set(expected_features) - set(input_data.columns)
            extra = set(input_data.columns) - set(expected_features)
            st.error(f"Feature mismatch! Missing: {missing}, Extra: {extra}")
        else:
            # Proceed with prediction
            prediction_proba = model.predict_proba(input_data[expected_features])[0][1]
            
        # Make prediction
        prediction_proba = model.predict_proba(input_data[expected_features])[0][1]
        prediction = 'Yes' if prediction_proba > 0.5 else 'No'
        
        # Show result
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Subscription Probability", 
                value=f"{prediction_proba:.1%}"
            )
        
        with col2:
            if prediction == 'Yes':
                st.success("This customer is likely to subscribe to a term deposit!")
            else:
                st.error("This customer is unlikely to subscribe to a term deposit.")
        
        # Feature importance visualization
        if st.checkbox("Show key factors influencing this prediction"):
            st.write("Top features that influenced this prediction:")
            st.write("1. Call duration (longer calls increase likelihood)")
            st.write("2. Previous successful outcome (success increases likelihood)")
            st.write("3. Job type (certain jobs like management have higher conversion)")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(str(e))