from pyspark.sql import SparkSession
# Create a Spark session
spark = SparkSession.builder.appName("APS").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()    

print("✅ Spark Session created successfully!")
# Check if Spark session is created
if spark:
    print("✅ Spark session is created successfully!")