from pyspark.sql import SparkSession
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.param.shared import Param, Params
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, ArrayType, FloatType
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark import keyword_only

class CustomLabelIndexer(Transformer):
    @keyword_only
    def __init__(self, mappings=None):
        super(CustomLabelIndexer, self).__init__()
        self.mappings = Param(self, "mappings", "dict of column: mapping_dict")
        self._setDefault(mappings={})
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, mappings=None):
        return self._set(mappings=mappings)

    def _transform(self, dataset):
        mappings = self.getOrDefault(self.mappings)
        for col_name, mapping_dict in mappings.items():
            index_udf = udf(lambda x: mapping_dict.get(x, -1), IntegerType())
            dataset = dataset.withColumn(f"{col_name}_indexed", index_udf(col(col_name)))
        return dataset.drop(*mappings.keys())

class ScaledFeatureExpander(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None, outputPrefix="_scaled", withMean=True, withStd=True):
        super(ScaledFeatureExpander, self).__init__()
        self.inputCols = inputCols
        self.outputPrefix = outputPrefix

    def _transform(self, df):
        assembler = VectorAssembler(inputCols=self.inputCols, outputCol="numeric_features")
        df = assembler.transform(df)

        scaler = StandardScaler(
            inputCol="numeric_features",
            outputCol="scaled_features",
            withMean=self.withMean,
            withStd=self.withStd
        )
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)

        def unpack_vector(v):
            return v.toArray().tolist() if v is not None else [None] * len(self.inputCols)

        unpack_udf = udf(unpack_vector, ArrayType(FloatType()))
        df = df.withColumn("scaled_array", unpack_udf(col("scaled_features")))

        for i, col_name in enumerate(self.inputCols):
            df = df.withColumn(col_name + self.outputPrefix, col("scaled_array").getItem(i))

        return df.drop("numeric_features", "scaled_features", "scaled_array", *self.inputCols)

def load_data(spark, train_path, test_path):
    train = spark.read.csv(train_path, header=True, inferSchema=True)
    test = spark.read.csv(test_path, header=True, inferSchema=True)
    return train, test

def preprocess_data(spark, train, test):
    numeric_variables = [
        "Age", "Flight Distance",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]
    categorical_variables_int = [
        'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness'
    ]
    categorical_variables_str = [
        "Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"
    ]
    custom_mappings = {
        "Gender": {'Female': 0, 'Male': 1},
        "Customer Type": {'Loyal Customer': 1, 'disloyal Customer': 0},
        "Type of Travel": {'Business travel': 1, 'Personal Travel': 0},
        "Class": {'Eco': 0, 'Eco Plus': 1, 'Business': 2},
        "satisfaction": {'neutral or dissatisfied': 0, 'satisfied': 1}
    }

    imputer = Imputer(
        inputCols=["Arrival Delay in Minutes"],
        outputCols=["Arrival Delay in Minutes"],
        strategy="median"
    )
    scaler_stage = ScaledFeatureExpander(inputCols=numeric_variables, outputPrefix="_scaled")
    custom_label_indexer = CustomLabelIndexer(mappings=custom_mappings)
    assembler = VectorAssembler(
        inputCols=categorical_variables_int + [
            "Age_scaled", "Flight Distance_scaled",
            "Departure Delay in Minutes_scaled", "Arrival Delay in Minutes_scaled",
            "Gender_indexed", "Customer Type_indexed",
            "Type of Travel_indexed", "Class_indexed"
        ],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[imputer, scaler_stage, custom_label_indexer, assembler])
    pipeline_model = pipeline.fit(train)
    train_processed = pipeline_model.transform(train)
    test_processed = pipeline_model.transform(test)
    return train_processed, test_processed, pipeline_model