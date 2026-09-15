
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
logi = joblib.load('logi.sav')

# Define the feature names, ensuring they match the training data
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 
                 'Vehicle_Age', 'Road_Condition_Score', 'Package_Weight', 
                 'Fuel_Efficiency', 'Warehouse_Processing_Time']

# Streamlit App Title
st.title('Delivery Delay Prediction App')

st.write("Enter the features below to predict if there will be a delivery delay.")

# Input widgets for each feature
input_data = {}
for feature in feature_names:
    if feature in ['Delivery_Distance', 'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time']:
        input_data[feature] = st.number_input(f'Enter {feature}', value=10.0, step=0.1)
    else:
        input_data[feature] = st.slider(f'Enter {feature}', min_value=0, max_value=10, value=1)

# Convert input data to a DataFrame, ensuring correct order and shape
input_df = pd.DataFrame([input_data])

# Make prediction when a button is clicked
if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_df)
    prediction_proba = logi.predict_proba(input_df)

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error(f"The model predicts: **Delay**")
    else:
        st.success(f"The model predicts: **No Delay**")
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")


# Save the app.py file
with open('app.py', 'w') as f:
    f.write(st.session_state.app_code)
