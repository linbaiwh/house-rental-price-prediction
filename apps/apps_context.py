from pathlib import Path
import pandas as pd
import joblib
import sys
import os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_folder = Path(__file__).resolve().parents[1] / 'data'
raw_data_folder = data_folder / 'raw'
feature_folder = data_folder / 'feature'
processed_folder = data_folder / 'processed'
model_folder = data_folder / 'model'

def read_review(processed=False, **readkwargs):
    if processed:
        return pd.read_csv(processed_folder / 'review_tokens.csv', **readkwargs)

    return pd.read_csv(raw_data_folder / 'reviews.csv', **readkwargs)

def save_to_feature(file_name, df=None, pic=None, **savekwargs):
    if df is not None:
        df.to_csv(feature_folder / f'{file_name}.csv', index=False, **savekwargs)
    if pic is not None:
        pic.savefig(feature_folder / f'{file_name}.png')

def save_to_processed(file_name, df, **savekwargs):
    df.to_csv(processed_folder / f'{file_name}.csv', index=True, **savekwargs)

def save_model(file_name, model):
    joblib.dump(model, model_folder / f'{file_name}.joblib')

def read_model(file_name):
    return joblib.load(model_folder / f'{file_name}.joblib')
