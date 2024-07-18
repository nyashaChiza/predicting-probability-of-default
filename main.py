import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load the pre-trained model and pipeline
with open("pipelines\data_processing_pipeline_20240710_170133.pkl", "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

with open("pipelines\data_processing_pipeline_20240710_170133.pkl", "rb") as model_file:
    model = pickle.load(model_file)

def preprocess_input(data):
    # Convert to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Pass data through the pipeline
    processed_data = pipeline.transform(df)
    
    return processed_data

def make_prediction(data):
    preprocessed_data = preprocess_input(data)
    prediction = model.predict(preprocessed_data)
    return prediction[0]

app = FastAPI()

class LoanApplication(BaseModel):
    gender: str
    disbursement_date: str
    currency: str
    country: str
    is_employed: bool
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    remaining_term: int
    salary: float
    marital_status: str
    loan_status: str

@app.post("/predict")
def predict(application: LoanApplication):
    try:
        input_data = application.dict()
        prediction = make_prediction(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
