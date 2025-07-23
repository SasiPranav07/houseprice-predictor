from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import sys

# Add src path to load model
sys.path.append(str(Path(__file__).parent / "src"))
from src.model_training import HousePriceModelTrainer
from src.utils import format_currency

app = Flask(__name__)

# Load model once on startup
model_trainer = HousePriceModelTrainer()
try:
    model_trainer.load_model()
except FileNotFoundError:
    model_trainer = None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model_trainer is None:
        return render_template("index.html", error="Model not found. Train it first.")

    try:
        # Default values
        df_sample = pd.read_csv("data/train.csv") if os.path.exists("data/train.csv") else pd.DataFrame()

        default_features = {
            col: df_sample[col].median() if df_sample[col].dtype in ['int64', 'float64'] else df_sample[col].mode()[0]
            for col in df_sample.columns if col != "SalePrice"
        }

        # Update with form data
        user_input = {
            'OverallQual': int(request.form['overall_qual']),
            'YearBuilt': int(request.form['year_built']),
            'GrLivArea': int(request.form['gr_liv_area']),
            'GarageCars': int(request.form['garage_cars']),
            'FullBath': int(request.form['full_bath']),
            'TotalBsmtSF': int(request.form['total_bsmt_sf']),
            'Neighborhood': request.form['neighborhood'],
            'ExterQual': request.form['exter_qual'],
            'KitchenQual': request.form['kitchen_qual'],
        }

        default_features.update(user_input)
        input_df = pd.DataFrame([default_features])

        # Predict
        price = model_trainer.predict(input_df)[0]
        price_str = format_currency(price)

        return render_template("index.html", prediction=price_str)

    except Exception as e:
        return render_template("index.html", error=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)