from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, StringIndexer, OneHotEncoder, PCA
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import Transformer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, ArrayType
from pyspark.sql import SparkSession

# Custom Transformer for Scaled Feature Expansion
class ScaledFeatureExpander(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None, outputPrefix="_scaled"):
        super(ScaledFeatureExpander, self).__init__()
        self.inputCols = Param(self, "inputCols", "Input columns", typeConverter=TypeConverters.toListString)
        self.outputPrefix = Param(self, "outputPrefix", "Prefix for scaled columns", typeConverter=TypeConverters.toString)
        self._setDefault(outputPrefix="_scaled")
        if inputCols is not None:
            self._set(inputCols=inputCols)
        self._set(outputPrefix=outputPrefix)

    def _transform(self, df):
        inputCols = self.getOrDefault(self.inputCols)
        outputPrefix = self.getOrDefault(self.outputPrefix)
        assembler = VectorAssembler(inputCols=inputCols, outputCol="numeric_features")
        df = assembler.transform(df)
        scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features")
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        def unpack_vector(v):
            return v.toArray().tolist() if v is not None else [None]*len(inputCols)
        unpack_udf = udf(unpack_vector, ArrayType(FloatType()))
        df = df.withColumn("scaled_array", unpack_udf(col("scaled_features")))
        for i, col_name in enumerate(inputCols):
            df = df.withColumn(col_name + outputPrefix, col("scaled_array").getItem(i))
        return df.drop("numeric_features", "scaled_features", "scaled_array", *inputCols)

# Build Pipeline
def build_pipeline():
    # Define categorical and numeric columns
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    numeric_cols = [
        "Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]
    
    # Imputer for missing values
    imputer = Imputer(
        inputCols=["Arrival Delay in Minutes"],
        outputCols=["Arrival Delay in Minutes"]
    ).setStrategy("median")
    
    # StringIndexer for categorical columns
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep")
        for col in categorical_cols + ["satisfaction"]
    ]
    
    # OneHotEncoder for indexed categorical columns
    encoder = OneHotEncoder(
        inputCols=[col + "_index" for col in categorical_cols],
        outputCols=[col + "_encoded" for col in categorical_cols]
    )
    
    # ScaledFeatureExpander for numeric columns
    scaler = ScaledFeatureExpander(inputCols=numeric_cols)
    
    # VectorAssembler for combining features
    assembler = VectorAssembler(
        inputCols=[col + "_encoded" for col in categorical_cols] + [col + "_scaled" for col in numeric_cols],
        outputCol="features"
    )
    
    # PCA for dimensionality reduction
    pca = PCA(k=10, inputCol="features", outputCol="pca_features")
    
    # Build pipeline
    pipeline = Pipeline(stages=[imputer] + indexers + [encoder, scaler, assembler, pca])
    return pipeline

if __name__ == "__main__":
    spark = SparkSession.builder.appName("PipelineBuilding").getOrCreate()
    pipeline = build_pipeline()
    print("Pipeline built successfully.")