Demand Forecast
A Python library for SKU-level demand forecasting using quantile regression, designed for optimizing inventory management and supply chain planning. The project implements an end-to-end pipeline for data preprocessing, feature engineering, model training, evaluation, and deployment.
Features

Data Preprocessing: Extracts sales data from orders and constructs a daily SKU-level dataset.
Feature Engineering: Generates rolling window features (e.g., average and quantile-based) and target variables for multiple forecast horizons (7, 14, 21 days).
Quantile Regression: Uses QuantileRegressor from scikit-learn to predict demand at different quantiles (0.1, 0.5, 0.9) for robust forecasting.
Evaluation Metrics: Computes quantile loss to assess model performance across multiple horizons.
Missed Profits Analysis: Calculates weekly missed profits and confidence intervals to evaluate the financial impact of forecasting errors.
Automated Pipeline: Orchestrates the entire workflow using ClearML for data processing, training, and deployment.

Tech Stack

Python 3.9+
pandas
numpy
scikit-learn
clearml
requests
joblib
tqdm
fire

Installation

Clone the repository:git clone https://github.com/Lebedinskiy1377/demand_forecast.git
cd demand_forecast


Install dependencies:pip install -r requirements.txt


(Optional) Set up ClearML for pipeline orchestration:
Install ClearML: pip install clearml
Configure ClearML credentials (see ClearML documentation).



Usage
Running the Pipeline
The pipeline processes data, trains a model, evaluates it, and saves the model for production use. Run the main script with default parameters:
python src/model/training.py --orders_url https://disk.yandex.ru/d/NUDMAdBMe9sbLw --model_path models/model.pkl

Example Workflow

Prepare Data:

Download orders data from a provided URL (e.g., Yandex Disk).
Extract daily sales data and generate features (e.g., rolling averages and quantiles) and targets (e.g., demand for the next 7, 14, 21 days).


Train and Evaluate Model:


from src.model.training import main

# Run the pipeline
main(
    orders_url="https://disk.yandex.ru/d/NUDMAdBMe9sbLw",
    model_path="models/model.pkl",
    debug=True  # Run in debug mode for faster execution
)


Evaluate Missed Profits:

from src.model.missed_profits import week_missed_profits, missed_profits_ci
import pandas as pd

# Load predictions and actuals
df = pd.read_csv("data/pred.csv")

# Calculate weekly missed profits
result = week_missed_profits(df, sales_col="qty", forecast_col="pred_7d_q50")
print(result)

# Estimate confidence intervals
ci = missed_profits_ci(result, missed_profits_col="missed_profits")
print(ci)

Example Input Data
The input data should be a CSV file with columns timestamp, sku_id, sku, price, and qty.
Example (data/orders.csv):
timestamp,sku_id,sku,price,qty
2023-01-01,1,ItemA,626.66,1
2023-01-01,2,ItemB,1016.57,1
2023-01-02,1,ItemA,626.66,3

Example Output

Predictions (data/pred.csv):

sku_id,day,pred_7d_q10,pred_7d_q50,pred_7d_q90,pred_14d_q10,pred_14d_q50,pred_14d_q90,pred_21d_q10,pred_21d_q50,pred_21d_q90
1,2023-12-01,2.1,3.5,5.2,4.0,6.5,8.9,6.2,9.0,12.3
2,2023-12-01,1.8,3.0,4.7,3.5,5.8,7.9,5.0,7.5,10.1


Missed Profits:

day,revenue,missed_profits
2023-12-03,1253.32,187.99
2023-12-10,1987.65,245.32

Project Structure
├── data/               # Datasets (e.g., orders.csv, features.csv, pred.csv)
├── models/             # Trained models (e.g., model.pkl)
├── src/                # Source code
│   ├── model/
│   │   ├── evaluate.py       # Quantile loss evaluation
│   │   ├── features.py       # Feature and target generation
│   │   ├── model.py          # Quantile regression model and train-test split
│   │   ├── training.py       # Pipeline for data processing, training, and deployment
│   │   ├── missed_profits.py # Missed profits calculation and confidence intervals
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

Requirements
Install dependencies using:
pip install pandas numpy scikit-learn clearml requests joblib tqdm fire

Notes

The pipeline uses ClearML for task orchestration. In debug mode (--debug True), it runs locally without creating ClearML tasks for faster execution.
The model predicts demand for new SKUs as zero, as specified in model.py.
Feature engineering includes rolling window statistics (mean and quantiles) for 7, 14, and 21 days, which can be customized in training.py.
The project is designed for retail demand forecasting but can be adapted for other time-series forecasting tasks.
