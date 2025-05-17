from pyspark.sql import SparkSession
# Create a Spark session
spark = SparkSession.builder.appName("APS").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()    

print("✅ Spark Session created successfully!")
print(f"Spark Version: {spark.version}")
print(f"Spark Master: {spark.sparkContext.master}")
print(f"Spark Application Name: {spark.sparkContext.appName}")
print(f"Spark Driver Memory: {spark.sparkContext._conf.get('spark.driver.memory')}")
print(f"Spark Executor Memory: {spark.sparkContext._conf.get('spark.executor.memory')}")
print(f"Spark Default Parallelism: {spark.sparkContext.defaultParallelism}")
print(f"Spark UI Web URL: {spark.sparkContext.uiWebUrl}")