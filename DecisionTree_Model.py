from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from Pipeline import build_pipeline

# Initialize Spark Session
spark = SparkSession.builder.appName("DecisionTreeModel").getOrCreate()

def train_decision_tree(train_df):
    # Build pipeline
    pipeline = build_pipeline()
    
    # Decision Tree
    dt = DecisionTreeClassifier(labelCol="satisfaction_index", featuresCol="pca_features")
    
    # Combine pipeline with model
    pipeline = Pipeline(stages=[pipeline, dt])
    
    # Parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10]) \
        .addGrid(dt.maxBins, [16, 32]) \
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
    model.write("/content/Tuned_model/DecisionTree")
    print("Decision Tree model saved to /content/Tuned_model/DecisionTree")
    
    return model

if __name__ == "__main__":
    # Load train dataset
    train = spark.read.csv(
        "/content/AirlinePassengerSatisfaction/train.csv",
        header=True,
        inferSchema=True
    )
    train_decision_tree(train)