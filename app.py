import os
import sys
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, lib_path)

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import udf
from pyspark.ml.tuning import CrossValidatorModel
from Model.data_processing import preprocess_data, load_data, CustomLabelIndexer, ScaledFeatureExpander
from Model.utils import count_rowncol, count_null, unique_values, variable_type

def main():
    st.title("🛬 Predict Airline Satisfaction")
    
    # Initialize Spark
    spark = SparkSession.builder.appName("AirlinePassengerSatisfaction").config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate()
    
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        pd_df = pd.read_csv(uploaded_file)
        spark_df = spark.createDataFrame(pd_df)
        
        # Drop unnecessary columns
        for col_name in ["_c0", "id"]:
            if col_name in spark_df.columns:
                spark_df = spark_df.drop(col_name)
        
        # Preprocess data
        _, processed_df, pipeline_model = preprocess_data(spark, spark_df, spark_df)
        
        # Load pre-trained model
        model = CrossValidatorModel.load("models/full_streamlit_pipeline")
        predictions = model.transform(processed_df).select("satisfaction_indexed", "prediction", "probability")
        
        st.success("✅ Predictions generated!")
        
        # Round probabilities
        @udf(ArrayType(StringType()))
        def round_probability(vec):
            if vec is not None:
                return [f"{float(x):.3f}" for x in vec]
            return None
        
        predictions = predictions.withColumn("probability_rounded", round_probability(col("probability")))
        df_clean = predictions.drop("probability").withColumnRenamed("probability_rounded", "probability")
        
        df_pd = df_clean.toPandas()
        st.dataframe(df_pd)
        
        # Pie chart
        st.title("Pie Chart Of Prediction Accuracy")
        import matplotlib.pyplot as plt
        
        df_pd["correct"] = df_pd["satisfaction_indexed"] == df_pd["prediction"]
        correct_count = df_pd["correct"].sum()
        incorrect_count = len(df_pd) - correct_count
        
        labels = ['Correct', 'Incorrect']
        sizes = [correct_count, incorrect_count]
        colors = ['#1f77b4', '#ff7f0e']
        explode = (0.05, 0)
        
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
        ax.axis('equal')
        st.pyplot(fig)
        
        csv = df_pd.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", data=csv, file_name="predictions.csv", mime="text/csv")
    
    spark.stop()

if __name__ == "__main__":
    main()