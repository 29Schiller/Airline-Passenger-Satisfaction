import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, sum as spark_sum, when, collect_list, collect_set
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("AirlinePassengerEDA").getOrCreate()

# Function to get number of rows and columns for each dataset
def count_rowncol(df):
    print(type(df), "\nNumber of Rows: ", df.count(), "\nNumber of Columns: ", len(df.columns))

# Function to get null values for each column
def count_null(df):
    null_counts = df.select([
        spark_sum(when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)).alias(c)
        for c in df.columns
    ])
    null_counts.show(truncate=False)

# Function to get unique values for each column
def unique_values(df):
    unique_dict = {}
    for col_name in df.columns:
        unique = df.select(collect_set(col_name).alias("unique_values")).collect()
        unique_dict[col_name] = unique[0]["unique_values"]
    return unique_dict

# Function to sort variable types
def variable_type(df):
    types_list = df.dtypes
    str_variables = []
    int_variables = []
    for i in types_list:
        if i[1] == "string" and i[0]:
            str_variables.append(i[0])
        else:
            int_variables.append(i[0])
    return str_variables, int_variables

# EDA Execution
def perform_eda(train_df, test_df):
    print("Train dataset:")
    count_rowncol(train_df)
    print("\nTest dataset:")
    count_rowncol(test_df)
    print("\nTrain Schema:")
    train_df.printSchema()
    
    print("\nNull Values in Train Dataset:")
    count_null(train_df)
    print("\nNull Values in Test Dataset:")
    count_null(test_df)
    
    print("\nUnique Values in Train Dataset:")
    for col_name, values in unique_values(train_df).items():
        print(f"{col_name}: {values}")
    
    print("\nVariable Types in Train Dataset:")
    str_vars, int_vars = variable_type(train_df)
    print("String Variables:", str_vars)
    print("Integer/Double Variables:", int_vars)

if __name__ == "__main__":
    # Load datasets
    train = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/train.csv",
        header=True,
        inferSchema=True
    )
    test = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/test.csv",
        header=True,
        inferSchema=True
    )
    perform_eda(train, test)