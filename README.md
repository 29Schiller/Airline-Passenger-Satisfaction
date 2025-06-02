# Airline Passenger Satisfaction Prediction
Course: Big Data Technologies

![Airline Passenger Satisfaction.png](https://github.com/29Schiller/Airline-Passenger-Satisfaction/blob/main/Airline%20Passenger.png)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12.3-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Jupyter_Notebook-7.0.8-F37626?style=for-the-badge&amp;logo=jupyter&amp;logoColor=white">
</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/29Schiller/Airline-Passenger-Satisfaction">
  </a>

<h3 align="center">Airline Passenger Satisfaction Predictor</h3>
    <a href="https://github.com/29Schiller/Airline-Passenger-Satisfaction" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <a href="https://www.overleaf.com/project/6836e32f160cf976dd35e7d3" target="_blank"><img src="https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white" alt="LaTeX"></a>
   <p align="center">
      This project presents a scalable system for predicting airline passenger satisfaction using big data techniques and machine learning. By leveraging PySpark’s distributed processing capabilities, the application processes high-dimensional flight data efficiently. The system incorporates a complete pipeline—from data acquisition and transformation to predictive modeling—optimized for performance and reliability. The insights generated aim to support airlines in tailoring customer service strategies and enhancing operational decision-making based on predictive analytics.
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## 1. **Introduction** 📋

<div align="center">
<img src="screenshots/Intro.gif" alt="">
</div>

<div style="text-align:justify">
This chapter outlines the methodological approach undertaken in developing a flight passenger satisfaction prediction system. The project employs a data-driven machine learning pipeline implemented using PySpark, a distributed data processing framework well-suited for large-scale computations. The primary objective of the system is to predict the likelihood of passenger satisfaction based on a variety of in-flight and service-related attributes, enabling stakeholders—particularly airlines—to make informed, data-backed decisions to improve customer service quality. The methodology is divided into four major components: use case analysis, architectural and model design, and implementation strategy. The application offers dual functionality, allowing users to submit individual passenger data for real-time predictions or upload structured datasets in CSV format to perform batch prediction and analysis
</div>

## 2. **Team Members** :couplekiss_man_man:

| Order |   Name  |  ID |  Roles | Contribution (%) |                   
| :---: |:---------------------:|:-----------:| :----------------------------------------------------------------: |:----------------:|
|   1   |    Nguyễn Minh Đạt    | ITDSIU22166 | Front-end | 25% |
|   2   |   Nguyễn Dư Nhân    | ITDSIU22140 | App  |25% |
|   3   | Phạm Hải Phú  | ITDSIU22179 | Model |25% |
|   4   | Dương Ngọc Phương Anh | ITDSIU22135 | Model |25% |

## 3. **Features** ✨
- Manual Input Prediction : Designed for end-users, this interface allows users to manually input key passenger and flight-related information (e.g., age, travel class, service ratings). Upon submission, the system processes the data and immediately returns a prediction indicating whether the
passenger is likely to be satisfied or not.
- CSV Prediction: Tailored for analysts or operational staff, this interface enables users to upload a structured CSV file containing multiple passenger
records. The system then performs batch processing to generate satisfaction predictions for each entry and returns a downloadable file with results, along with summary visualizations.

## 4. **Requirement** :dart:
Before running the project, ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Pyspark 
- Hadoop

## 5. **Project Structure** 📂
```
Airline_Passenger_Satisfaction-main/
├── Model/                             # Contains scripts for data processing, model training, and utilities.
├── Build_App/                         # Contains the Streamlit app for predictions.
├── Dataset/                           # Stores `train.csv` and `test.csv`.
├── Colab Notebook + Report + Slide    # Contains Source for the project Big Data HCMIU.
├── requirements.txt                   # Documents required package versions.
├── README.md/                         # ReadMe file of repository
└── .idea/                             # IDE configuration (optional)
```

## 6.Setup
1. **Clone the repository**:
   ```bash
   git clone <https://github.com/29Schiller/Airline-Passenger-Satisfaction>
   cd AirlinePassengerSatisfaction
   ```
2. **Install Java** (required for PySpark):
   - Ensure Java 8 or 11 is installed (e.g., `openjdk-11-jdk` on Linux or equivalent on Windows/macOS).
3. **Download data**:
   - Download `AirlinePassengerSatisfaction.zip` and unzip it to the `data/` folder to get `train.csv` and `test.csv`.
   - In this repository, we have already downloaded and unziped the dataset for you.
4. **Run file by order**:
   - Run the Data_preprocess for a cleaned data.
   - Run the Pipeline.py for building the pipeline
   - Run python file for each model: LR, RF, MLP and GBT.
   - Run the App.py for the app in streamlit.


## 7. Training Models
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

## 8. Acknowledgement <a name="Acknowledgement"></a> :brain:
<div style="text-align:justify">
We express our sincere gratitude and appreciation to Dr. Ho Long Van for his professional guidance. His unwavering encouragement and support were instrumental in helping our team achieve its goals.

We would also like to express our sincere gratitude to the irreplaceable members of our group. Their technical expertise and collaborative spirit were essential to our progress. Beyond their willingness to share their knowledge and troubleshoot challenges, their good humor and positive attitudes made this project an enriching and enjoyable learning experience. We are grateful to have had the opportunity to work alongside such a talented and supportive team.
</div>

## 9. References <a name="References">:bookmark:
- [Kaggle: Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- [StandardScaler, MinMaxScaler and RobustScaler technique](https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/)
- [Multilayer Perceptron (MLP)](https://spotintelligence.com/2024/02/20/multilayer-perceptron-mlp/)
- [Random Forest Algorithm:](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)
- [Gradient Boosting Algorithm:)](https://www.geeksforgeeks.org/ml-gradient-boosting/)
- [Logistics Regression Algorithm:](https://www.geeksforgeeks.org/understanding-logistic-regression/)
