import os
import sys
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, lib_path)

from pyspark.sql.functions import col, sum as spark_sum, when, collect_set
from pyspark.sql import SparkSession

def count_rowncol(df):
    print(type(df), "\nNumber of Rows: ", df.count(), "\nNumber of Columns: ", len(df.columns))

def count_null(df):
    null_counts = df.select([
        spark_sum(when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)).alias(c)
        for c in df.columns
    ])
    null_counts.show(truncate=False)

def unique_values(df):
    unique_dict = {}
    for col_name in df.columns:
        unique = df.select(collect_set(col_name).alias("unique_values")).collect()
        unique_dict[col_name] = unique[0]["unique_values"]
    return unique_dict

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