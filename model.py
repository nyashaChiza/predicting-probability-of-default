# model.py
import pickle
import pandas as pd
# Load the pre-trained model
with open('models/loan_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(X: pd.DataFrame):
    y_pred = model.predict(X)
    y_conf = model.predict_proba(X)[:, 1]
    return {'predition':y_pred, 'confidence':y_conf}