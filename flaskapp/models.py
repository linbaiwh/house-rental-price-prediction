from pathlib import Path
import joblib


data_folder = Path(__file__).resolve().parents[1] / 'data'
model_folder = data_folder / 'model'

rf_model = joblib.load(model_folder / 'rf_price.joblib')