# Signal Research for Indian Equities  
**An end-to-end FastAPI pipeline for SARIMA + LSTM-based price forecasting on NSE/BSE stocks**

---

## Overview

This project is an **AI-driven market signal research engine** built to forecast stock prices for **any NSE/BSE-listed equity** using both **classical (SARIMA)** and **deep learning (LSTM)** models.  
It features automated data ingestion from Yahoo Finance, feature engineering, model training, and next-day forecasting — all exposed via a clean **FastAPI interface**.

---

## Tech Stack

| Layer | Tools / Frameworks |
|-------|--------------------|
| **Data Source** | YahooQuery (Live OHLCV data from NSE/BSE) |
| **Feature Engineering** | Pandas, NumPy |
| **Classical Model** | SARIMA (Statsmodels) |
| **Deep Learning Model** | LSTM (TensorFlow / Keras) |
| **Backend API** | FastAPI, Uvicorn |
| **Storage** | Parquet, Pickle, Keras SavedModel |
| **Logging & Config** | Custom Logger + YAML Config Loader |

---
# Setup and Run
## 1. Clone the repository
git clone https://github.com/RohanGandhi980/Signal_Research_Indian_Equities.git
cd Signal_Research_Indian_Equities

## 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Start FastAPI server
uvicorn src.api.app:app --reload

---
# Example Outputs
RELIANCE.NS → 1-day forecast: ₹2850.31

TCS.NS → 1-day forecast: ₹4168.83

TECHM.NS → 1-day forecast: ₹1462.81


