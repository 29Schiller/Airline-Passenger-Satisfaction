from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("AirlinePassengerPreprocessing").getOrCreate()

# Function to check null values
def check_null_values(df, dataset_name):
    print(f"\nNull Values in {dataset_name} Dataset:")
    null_counts = df.select([
        spark_sum(when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)).alias(c)
        for c in df.columns
    ])
    null_counts.show(truncate=False)
    return null_counts

# Function to check outliers using IQR
def check_outliers(df, numeric_cols):
    outliers_info = {}
    for col_name in numeric_cols:
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        Q1, Q3 = quantiles[0], quantiles[1]
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()
        outliers_info[col_name] = {"lower_bound": lower_bound, "upper_bound": upper_bound, "outlier_count": outliers}
    return outliers_info

# Preprocessing function
def preprocess_data(train_df, output_path="Cleaned_train.csv"):
    # Check null values
    check_null_values(train_df, "Train")
    
    # Numeric columns for outlier detection
    numeric_cols = [
        "Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]
    
    # Check outliers
    outliers_info = check_outliers(train_df, numeric_cols)
    print("\nOutlier Information:")
    for col_name, info in outliers_info.items():
        print(f"{col_name}: {info}")
    
    # Handle missing values (imputation for 'Arrival Delay in Minutes')
    imputer = Imputer(inputCols=["Arrival Delay in Minutes"], outputCols=["Arrival Delay in Minutes"]).setStrategy("median")
    train_df = imputer.fit(train_df).transform(train_df)
    
    # Save cleaned dataset
    train_df.toPandas().to_csv(output_path, index=False)
    print(f"\nCleaned train dataset saved to {output_path}")

if __name__ == "__main__":
    # Load train dataset
    train = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/train.csv",
        header=True,
        inferSchema=True
    )
    preprocess_data(train)