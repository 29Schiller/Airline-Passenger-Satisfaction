import os
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

# Initialize a Spark Session
spark = SparkSession\
    .builder\
    .appName("AirlinePassengerSatisfaction")\
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()

# Check if the Spark Session is active
if 'spark' in locals() and isinstance(spark, SparkSession):
    print("SparkSession is active and ready to use.")
else:
    print("SparkSession is not active. Please create a SparkSession.")

# Load the dataset
# Load train file
train = spark.read.csv(
    "Dataset\\train.csv",
    header=True,
    inferSchema=True
)
# Load test file
test = spark.read.csv(
    "Dataset\\test.csv",
    header=True,
    inferSchema=True
)

train.show(5, truncate=False)
test.show(5, truncate=False)

# Define a function to find the best imputation strategy for a column
def find_best_imputation_strategy(df, col_name, strategy_options=["mean", "median"]):
    """
    Finds the best imputation strategy (mean, median, or mode) for a given column
    by evaluating which one minimizes the standard deviation of the column after imputation.
    This is a heuristic and might not be the best strategy for all scenarios.
    """
    original_std = df.select(col_name).agg({'`{}`'.format(col_name): 'stddev'}).collect()[0][0]
    print(f"Original standard deviation of '{col_name}': {original_std}")
    best_strategy = None
    min_std_diff = float('inf')

    for strategy in strategy_options:
        imputer = Imputer(inputCols=[col_name], outputCols=[col_name], strategy=strategy)
        try:
            model = imputer.fit(df)
            df_imputed = model.transform(df)
            imputed_std = df_imputed.select(col_name).agg({'`{}`'.format(col_name): 'stddev'}).collect()[0][0]
            #print(f"Standard deviation after imputation with '{strategy}': {imputed_std}")

            if imputed_std is not None and original_std is not None:
                std_diff = abs(imputed_std - original_std)
                if std_diff < min_std_diff:
                    min_std_diff = std_diff
                    #print(f"New minimum standard deviation difference: {min_std_diff}")
                    best_strategy = strategy
        except Exception as e:
            print(f"Could not apply strategy '{strategy}' to column '{col_name}': {e}")
            continue

    return best_strategy

# Checking null values of Train dataset
print("The null values in Train dataset:")
count_null(train)

# Columns with null values identified from previous count_null output
columns_to_impute = ['Arrival Delay in Minutes']

# Impute null values in the train dataset
print("\nImputing null values in the train dataset...")
for col_name in columns_to_impute:
    best_strategy_train = find_best_imputation_strategy(train, col_name, strategy_options=["mean", "median"])
    if best_strategy_train:
        print(f"Best imputation strategy for '{col_name}' in train: {best_strategy_train}")
        imputer = Imputer(inputCols=[col_name], outputCols=[col_name], strategy=best_strategy_train)
        model = imputer.fit(train)
        train = model.transform(train)
    else:
        print(f"Could not find a suitable strategy for '{col_name}' in train.")

print("\nChecking null values of Train dataset after imputation:")
count_null(train)


# Function to detect and replace outliers with the mean using IQR
def check_outlier(df, col_name):
    print(f"\nProcessing column: {col_name}")
    # Calculate Q1, Q3, and IQR
    quartiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    Q1 = quartiles[0]
    Q3 = quartiles[1]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"  Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"  Lower Bound (IQR method): {lower_bound}")
    print(f"  Upper Bound (IQR method): {upper_bound}")

    # Identify outliers
    is_outlier_col = (col(col_name) < lower_bound) | (col(col_name) > upper_bound)
    outliers_count = df.filter(is_outlier_col).count()
    total_count = df.count()
    print(f"  Number of outliers detected: {outliers_count} ({outliers_count/total_count:.2%})")

    return df

numeric_columns_for_outlier_check = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
categorical_variables_str = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                             'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                             'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
                             'Cleanliness']

total_column = numeric_columns_for_outlier_check + categorical_variables_str

# Apply outlier replacement to the selected numerical columns in the training data
for col_name in total_column:
    train = check_outlier(train, col_name)