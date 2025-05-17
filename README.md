# Airline Passenger Satisfaction Prediction

This project predicts airline passenger satisfaction using PySpark for data processing and model training, and Streamlit for a web-based prediction interface. All required libraries are bundled in the `lib` folder.

## Project Structure
- `Model/`: Contains scripts for data processing, model training, and utilities.
- `Build App/`: Contains the Streamlit app for predictions.
- `data/`: Stores `train.csv` and `test.csv`.
- `lib/`: Contains wheel files for all required Python packages.
- `requirements.txt`: Documents required package versions.
- `setup.py`: Defines the project as a Python package.
- `README.md`: Project documentation.

## Setup
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd AirlinePassengerSatisfaction
   ```
2. **Install Java** (required for PySpark):
   - Ensure Java 8 or 11 is installed (e.g., `openjdk-11-jdk` on Linux or equivalent on Windows/macOS).
3. **Download data**:
   - Download `AirlinePassengerSatisfaction.zip` and unzip it to the `data/` folder to get `train.csv` and `test.csv`.
4. **Place pre-trained model**:
   - Ensure the `full_streamlit_pipeline` folder is in `Build App/models/`.
5. **Verify `lib` folder**:
   - The `lib` folder contains all required `.whl` files (e.g., `pyspark-3.5.1-py2.py3-none-any.whl`, `pandas-2.2.2-cp311-cp311-win_amd64.whl`, etc.).
   - If missing, download wheels using:
     ```bash
     pip download --dest lib --platform win_amd64 --python-version 3.11 <package_name>==<version>
     ```

## Running the App
1. Navigate to the project directory:
   ```bash
   cd Build\ App
   ```
2. Run the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```
3. Access the app in your browser at `http://localhost:8501`.

## Training Models
To train models, use scripts in the `Model` folder. Example:
```python
from pyspark.sql import SparkSession
from Model.data_processing import load_data, preprocess_data
from Model.model_training import train_logistic_regression, evaluate_model

spark = SparkSession.builder.appName("TrainModel").getOrCreate()
train, test = load_data(spark, "data/train.csv", "data/test.csv")
train_processed, test_processed, _ = preprocess_data(spark, train, test)
model = train_logistic_regression(train_processed)
evaluate_model(model, test_processed, "Logistic Regression")
spark.stop()
```

## Notes
- The project uses libraries from the `lib` folder, added to `sys.path` at runtime.
- Ensure the wheel files in `lib` match your platform and Python version (e.g., `cp311-cp311-win_amd64` for Python 3.11 on Windows).
- The pre-trained model (`full_streamlit_pipeline`) must be in `Build App/models/`. To retrain, use `train_tuned_logistic_regression` and save it:
  ```python
  cv_model.write("Build App/models/full_streamlit_pipeline")
  ```
- PySpark requires Java. Set the `JAVA_HOME` environment variable if needed.