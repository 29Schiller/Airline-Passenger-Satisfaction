import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, ArrayType
from pyspark.ml.param.shared import Param, Params, TypeConverters

# Start Spark
spark = SparkSession.builder.appName("CSV Prediction").getOrCreate()

# Custom Transformer
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

# Define the pages
demo_1 = st.Page("/content/page1.py", title="Manual Input Prediction", icon="📊")
demo_2 = st.Page("/content/page2.py", title="Upload CSV Prediction", icon="📈")

# Set up navigation
pg = st.navigation([demo_1, demo_2])

# Run the selected page
pg.run()