import os
import sys
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, lib_path)

from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import rand

def train_logistic_regression(train):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="satisfaction_indexed",
        maxIter=500,
        regParam=0.001,
        elasticNetParam=1.0,
        family="binomial"
    )
    lr_model = lr.fit(train)
    return lr_model

def train_mlp(train):
    input_size = len(train.select("features").first()[0])
    layers = [input_size, 10, 2]
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="satisfaction_indexed",
        maxIter=100,
        layers=layers,
        blockSize=128,
        seed=123
    )
    mlp_model = mlp.fit(train)
    return mlp_model

def train_tuned_logistic_regression(train):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="satisfaction_indexed",
        family="binomial"
    )
    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.001, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(
        labelCol="satisfaction_indexed",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=2,
        seed=42
    )
    cv_model = cv.fit(train)
    return cv_model

def evaluate_model(model, test, model_name):
    predictions = model.transform(test)
    evaluator = BinaryClassificationEvaluator(
        labelCol="satisfaction_indexed",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    print(f"{model_name} AUC: {auc:.4f}")
    predictions.orderBy(rand(seed=42)).select(
        "satisfaction_indexed", "prediction", "probability"
    ).show(10, truncate=False)
    return predictions