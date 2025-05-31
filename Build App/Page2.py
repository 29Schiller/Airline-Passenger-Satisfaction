import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel

# Initialize Spark Session
spark = SparkSession.builder.appName("CSVPrediction").getOrCreate()

# Streamlit Page
st.set_page_config(page_title="Airline Satisfaction Prediction", page_icon="✈️", layout="wide")

# Header
st.markdown("""
<div style='text-align: center;'>
    <h1>✈️ Airline Satisfaction Prediction System</h1>
    <h3>Upload CSV for Prediction</h3>
</div>
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    df_pd = pd.read_csv(uploaded_file)
    df = spark.createDataFrame(df_pd)
    
    # Load Model
    try:
        model = CrossValidatorModel.read().load("/content/Tuned_model/LogisticRegression")
    except:
        st.error("Model not found. Please ensure the model is saved at /content/Tuned_model/LogisticRegression")
        st.stop()
    
    # Make Predictions
    predictions = model.transform(df)
    
    # Convert to Pandas for EDA
    numeric_columns = [
        "Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]
    categorical_columns = ["Gender", "Customer Type", "Type of Travel", "Class"]
    
    df_1 = df.select("id", *categorical_columns, *numeric_columns)
    df_2 = predictions.select("id", "prediction")
    
    df_EDA = df_1.join(df_2, on="id", how="inner").drop("id")
    df_EDA = df_EDA.toPandas()
    
    st.markdown("<h2 style='text-align:center;'>📊 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    
    # Service Rating Differences
    st.subheader("📋 What Services Affect Satisfaction Most")
    
    service_columns = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness'
    ]
    
    service_summary = df_EDA.groupby('prediction')[service_columns].mean().T
    service_summary['Difference'] = service_summary[1.0] - service_summary[0.0]
    service_summary_sorted = service_summary.sort_values(by='Difference', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=service_summary_sorted.index, x=service_summary_sorted['Difference'], hue=service_summary_sorted.index, palette='coolwarm', ax=ax2, legend=False)
    ax2.axvline(0, color='gray', linestyle='--')
    ax2.set_title('Average Service Rating Difference: Satisfied - Dissatisfied')
    ax2.set_xlabel('Rating Difference')
    ax2.set_ylabel('Service Category')
    st.pyplot(fig2)
    
    df_EDA['prediction'] = df_EDA['prediction'].replace({
        0.0: 'Neutral or Dissatisfied',
        1.0: 'Satisfied'
    })
    
    # Customer Profile
    st.subheader("👥 Satisfaction by Customer Profile")
    
    profile_vars = ['Gender', 'Customer Type', 'Class', 'Type of Travel']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, var in zip(axes.flat, profile_vars):
        sns.countplot(data=df_EDA, x=var, hue='prediction', ax=ax)
        ax.set_title(f'Satisfaction by {var}')
        ax.set_ylabel("Passenger Count")
        ax.set_xlabel(var)
        ax.tick_params(axis='x', rotation=0)
    
    st.pyplot(fig)
    
    # Delay Analysis
    st.subheader("⏱️ Delay Impact on Satisfaction")
    
    delay_summary = df_EDA.groupby('prediction')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean()
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    delay_summary.T.plot(kind='bar', ax=ax3, colormap='viridis')
    ax3.set_title('Average Delay by Satisfaction Group')
    ax3.set_ylabel('Average Delay (minutes)')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    st.pyplot(fig3)
    
    # Download Results
    csv = df_pd.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", data=csv, file_name="predictions.csv", mime="text/csv")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 2rem; color: #64748B; font-size: 0.9rem;'>
    <p>Powered by Streamlit & PySpark | ✈️ Airline Satisfaction Prediction System</p>
    <p>Big Data Technology Project - Hoi Dong O5 - International University (IU), Vietnam National University – Ho Chi Minh City</p>
</div>
""", unsafe_allow_html=True)