from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from Pipeline import build_pipeline

# Initialize Spark Session
spark = SparkSession.builder.appName("LogisticRegressionModel").getOrCreate()

def train_logistic_regression(train_df):
    # Build pipeline
    pipeline = build_pipeline()
    
    # Logistic Regression
    lr = LogisticRegression(labelCol="satisfaction_index", featuresCol="pca_features")
    
    # Combine pipeline with model
    pipeline = Pipeline(stages=[pipeline, lr])
    
    # Parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
        .build()
    
    # Evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="satisfaction_index", metricName="areaUnderROC")
    
    # CrossValidator
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3
    )
    
    # Fit model
    model = cv.fit(train_df)
    
    # Save model
    model.write("/content/Tuned_model/LogisticRegression")
    print("Logistic Regression model saved to /content/Tuned_model/LogisticRegression")
    
    return model

if __name__ == "__main__":
    # Load train dataset
    train = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/train.csv",
        header=True,
        inferSchema=True
    )
    train_logistic_regression(train)