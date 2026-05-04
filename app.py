import streamlit as st
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load all the encoders and scaler
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title('🏦 Customer Churn Prediction')

# Create input form
with st.form("prediction_form"):
    st.header("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        age = st.slider("Age", min_value=18, max_value=100, value=40)
        tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
        credit_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        country = st.selectbox("Country", one_hot_encoder.categories_[0])
        gender = st.selectbox("Gender", label_encoder_gender.classes_)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
        active_member = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    products_number = st.slider("Number of Products", min_value=1, max_value=4, value=2)
    estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, value=100000.0)

    submit_button = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# Perform prediction when form is submitted
if submit_button:
    # Create input dataframe
    input_data = pd.DataFrame({
        'credit_score': [credit_score],
        'country': [country],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary],
    })

    # Encode gender (convert string to numeric)
    input_data['gender'] = label_encoder_gender.transform(input_data['gender'].values)

    # One-hot encode country
    geo_encoded = one_hot_encoder.transform([[country]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder.get_feature_names_out(['country'])
    )

    # Drop country column and concatenate with geo-encoded data
    input_data = input_data.drop('country', axis=1)
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the features (all numeric now, matching training data)
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = prediction[0][0]

    # Display result
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction_proba > 0.5:
            st.error(f'🔴 Customer will CHURN')
            st.metric("Churn Risk", f"{prediction_proba:.1%}", delta="High Risk")
        else:
            st.success(f'🟢 Customer will NOT churn')
            st.metric("Retention Rate", f"{1-prediction_proba:.1%}", delta="Low Risk")

    with col2:
        st.write("### Churn Probability Distribution")
        st.progress(float(prediction_proba))
        st.write(f"**Churn Probability:** {prediction_proba:.2%}")
        st.write(f"**Retention Probability:** {1-prediction_proba:.2%}")
































