# 📈 Demand Forecasting System for Retail (Quantile Regression + ClearML)

This project implements a **multi-horizon, multi-quantile demand forecasting pipeline** for retail sales using `QuantileRegressor`.  
It includes **automated feature generation**, **ClearML pipelines**, **quantile-based evaluation**, and a **business-focused metric: missed profits**.

---

## Problem Statement

Forecast demand for each SKU for the next 7, 14, and 21 days using past sales history.  
Key goals:
- Quantify uncertainty (quantile regression)
- Generate interpretable features
- Evaluate using business-impact metrics (missed profits)

---

## Key Features

✅ Fully automated ML pipeline (using `ClearML`)  
✅ Custom quantile loss + bootstrap confidence intervals  
✅ Rolling feature generation (quantile, average)  
✅ Per-SKU models trained in parallel  
✅ Business metric: **missed profit** estimation  
✅ Modular codebase, easy to extend or adapt

---

## Pipeline Overview

Raw orders (.csv or Yandex.Disk) → Feature engineering → Targets → Train/Test split →  
Per-SKU QuantileRegressor training → Predictions → Evaluation (quantile loss, missed profits) → Model saved

```python
features = {
    "qty_7d_avg": ("qty", 7, "avg", None),
    "qty_14d_q50": ("qty", 14, "quantile", 50),
    ...
}

targets = {
    "next_7d": ("qty", 7),
    "next_14d": ("qty", 14),
    "next_21d": ("qty", 21),
}
```

## Project Structure

├── src/
│   ├── model/
│   │   ├── model.py         # QuantileRegressor logic + training
│   │   ├── features.py      # Rolling feature & target generation
│   │   ├── evaluate.py      # Quantile loss evaluation
│   │   └── training.py      # ClearML pipeline components
│
├── test/
│   └── missed_profit.py     # Business metric: missed profit + CI
├── data/                    # Input/output data
├── models/                  # Trained models
├── README.md                # You’re here

## Evaluation Example

```python
from evaluate import evaluate_model

losses = evaluate_model(df_test, df_pred, quantiles=[0.1, 0.5, 0.9], horizons=[7, 14, 21])
print(losses)
```

## Business Metric: Missed Profits

`test/missed_profit.py` includes:

- Weekly missed revenue due to underforecasting
    
- Confidence intervals via bootstrapping
    
- Relative loss (% of total revenue)

## Tech Stack

- 🧠 `sklearn.linear_model.QuantileRegressor`
    
- ⚡ `joblib` for parallel model training
    
- 📈 `ClearML` for orchestrating training pipeline
    
- 📊 `Pandas`, `NumPy`, `TQDM`, `Fire`
    
- 📦 Lightweight, no deep learning dependencies

## Launch Pipeline (Yandex.Disk Example)

```bash
python training.py --debug=False
```

## Author

Dmitry Lebedinskiy (2024)  
Developed as part of demand forecasting R&D track for retail ML systems.
