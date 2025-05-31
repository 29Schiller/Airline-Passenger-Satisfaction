from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from Pipeline import build_pipeline

# Initialize Spark Session
spark = SparkSession.builder.appName("RandomForestModel").getOrCreate()

def train_random_forest(train_df):
    # Build pipeline
    pipeline = build_pipeline()
    
    # Random Forest
    rf = RandomForestClassifier(labelCol="satisfaction_index", featuresCol="pca_features")
    
    # Combine pipeline with model
    pipeline = Pipeline(stages=[pipeline, rf])
    
    # Parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
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
    model.write("/content/Tuned_model/RandomForest")
    print("Random Forest model saved to /content/Tuned_model/RandomForest")
    
    return model

if __name__ == "__main__":
    # Load train dataset
    train = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/train.csv",
        header=True,
        inferSchema=True
    )
    train_random_forest(train)