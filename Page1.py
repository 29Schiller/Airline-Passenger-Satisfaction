import streamlit as st
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# Initialize Spark Session
spark = SparkSession.builder.appName("ManualInputPrediction").getOrCreate()

# Streamlit Page
st.set_page_config(page_title="Airline Satisfaction Prediction", page_icon="✈️", layout="wide")

# Header
st.markdown("""
<div style='text-align: center;'>
    <h1>✈️ Airline Satisfaction Prediction System</h1>
    <h3>Manual Input Prediction</h3>
</div>
""", unsafe_allow_html=True)

# Load Model
try:
    model = CrossValidatorModel.read().load("/content/Tuned_model/LogisticRegression")
except:
    st.error("Model not found. Please ensure the model is saved at /content/Tuned_model/LogisticRegression")
    st.stop()

# Input Form
with st.form("passenger_form"):
    st.subheader("Enter Passenger Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        flight_distance = st.number_input("Flight Distance (km)", min_value=0, value=500)
        departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, value=0)
        arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0.0, value=0.0)
    
    with col3:
        wifi = st.slider("Inflight Wifi Service (0-5)", 0, 5, 3)
        departure_arrival_time = st.slider("Departure/Arrival Time Convenient (0-5)", 0, 5, 3)
        online_booking = st.slider("Ease of Online Booking (0-5)", 0, 5, 3)
        gate_location = st.slider("Gate Location (0-5)", 0, 5, 3)
        food_drink = st.slider("Food and Drink (0-5)", 0, 5, 3)
        online_boarding = st.slider("Online Boarding (0-5)", 0, 5, 3)
        seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 3)
        entertainment = st.slider("Inflight Entertainment (0-5)", 0, 5, 3)
        onboard_service = st.slider("On-board Service (0-5)", 0, 5, 3)
        leg_room = st.slider("Leg Room Service (0-5)", 0, 5, 3)
        baggage_handling = st.slider("Baggage Handling (0-5)", 0, 5, 3)
        checkin_service = st.slider("Checkin Service (0-5)", 0, 5, 3)
        inflight_service = st.slider("Inflight Service (0-5)", 0, 5, 3)
        cleanliness = st.slider("Cleanliness (0-5)", 0, 5, 3)
    
    submitted = st.form_submit_button("Predict Satisfaction")
    
    if submitted:
        # Create DataFrame
        data = {
            "Gender": [gender], "Customer Type": [customer_type], "Type of Travel": [travel_type], "Class": [travel_class],
            "Age": [age], "Flight Distance": [flight_distance], "Inflight wifi service": [wifi],
            "Departure/Arrival time convenient": [departure_arrival_time], "Ease of Online booking": [online_booking],
            "Gate location": [gate_location], "Food and drink": [food_drink], "Online boarding": [online_boarding],
            "Seat comfort": [seat_comfort], "Inflight entertainment": [entertainment], "On-board service": [onboard_service],
            "Leg room service": [leg_room], "Baggage handling": [baggage_handling], "Checkin service": [checkin_service],
            "Inflight service": [inflight_service], "Cleanliness": [cleanliness],
            "Departure Delay in Minutes": [departure_delay], "Arrival Delay in Minutes": [arrival_delay]
        }
        df = spark.createDataFrame(pd.DataFrame(data))
        
        # Make Prediction
        prediction = model.transform(df)
        prediction = prediction.select("prediction").collect()[0][0]
        result = "Satisfied" if prediction == 1.0 else "Neutral or Dissatisfied"
        
        # Display Result
        st.markdown(f"<h3 style='text-align: center; color: {'green' if prediction == 1.0 else 'red'}'>Predicted Satisfaction: {result}</h3>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 2rem; color: #64748B; font-size: 0.9rem;'>
    <p>Powered by Streamlit & PySpark | ✈️ Airline Satisfaction Prediction System</p>
    <p>Big Data Technology Project - Hoi Dong O5 - International University (IU), Vietnam National University – Ho Chi Minh City</p>
</div>
""", unsafe_allow_html=True)