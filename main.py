import dill
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.transformers import age_bin_transformer, salary_bin_transformer, bin_age, bin_salary



# Load the pre-trained model and pipeline

pipeline_path ="pipelines/data_processing_pipeline_20240720_184845.pkl"
with open(pipeline_path, 'rb') as f:
    pipeline = dill.load(f)
        
model_path = "models/classification_model.pkl"
model = joblib.load(model_path)

def preprocess_input(data):
    if 1:
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(data, index=[0])
        
        # Apply custom transformations
        # df['age'] = bin_age(df['age'])
        # df['salary'] = bin_salary(df['salary'])

        # Pass data through the pipeline
        pipeline.fit(df)
        processed_data = pipeline.transform(df)
        input(processed_data)
        return processed_data
    else:#except Exception as e:
        1#raise ValueError(f"Error in preprocessing input data: {e}")

def make_prediction(data):
    if 1:
        preprocessed_data = preprocess_input(data)
        input(preprocessed_data)
        prediction = model.predict(preprocessed_data)
        return prediction[0]
    else:#except Exception as e:
        1#raise ValueError(f"Error in making prediction: {e}")

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

@app.post("/test")
def test(data: dict):
    return data
@app.post("/predict")
def predict(application: LoanApplication):
    if 1:
        input(application.dict())
        input_data = application.dict()
        prediction = make_prediction(input_data)
        return {"prediction": prediction}
    else:#except Exception as e:
        1#raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
