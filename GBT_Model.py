from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from Pipeline import build_pipeline

# Initialize Spark Session
spark = SparkSession.builder.appName("GBTModel").getOrCreate()

def train_gbt(train_df):
    # Build pipeline
    pipeline = build_pipeline()
    
    # Gradient Boosted Tree
    gbt = GBTClassifier(labelCol="satisfaction_index", featuresCol="pca_features")
    
    # Combine pipeline with model
    pipeline = Pipeline(stages=[pipeline, gbt])
    
    # Parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [10, 20]) \
        .addGrid(gbt.maxDepth, [5, 10]) \
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
    model.write("/content/Tuned_model/GBT")
    print("GBT model saved to /content/Tuned_model/GBT")
    
    return model

if __name__ == "__main__":
    # Load train dataset
    train = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/train.csv",
        header=True,
        inferSchema=True
    )
    train_gbt(train)