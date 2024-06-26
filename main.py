from typing import Union, List
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

app = FastAPI()

# Load the pre-trained model
with open('models/loan_classification_model.pkl', 'rb') as f:
    model = ''#pickle.load(f)

# Create the SQLite database engine
engine = create_engine('sqlite:///loans.db')

# Define the SQLAlchemy model
Base = declarative_base()

class LoanModel(Base):
    __tablename__ = 'loans'

    id = Column(Integer, primary_key=True)
    gender = Column(String)
    disbursement_date = Column(String)
    currency = Column(String)
    country = Column(String)
    is_employed = Column(Boolean)
    job = Column(String)
    location = Column(String)
    loan_amount = Column(Float)
    number_of_defaults = Column(Integer)
    outstanding_balance = Column(Float)
    interest_rate = Column(Float)
    age = Column(Integer)
    remaining_term = Column(Integer)
    salary = Column(Float)
    marital_status = Column(String)
    loan_status = Column(String)
    target = Column(Boolean)
    predicted_score = Column(Float)
    predicted_confidence = Column(Float)

# Create the database table if it doesn't exist
Base.metadata.create_all(engine)

class LoanData(BaseModel):
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
    target: bool

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/loan_data/{loan_id}")
def read_loan_data(loan_id: int):
    with engine.connect() as conn:
        loan = conn.execute(
            LoanModel.select().where(LoanModel.id == loan_id)
        ).one_or_none()
        if loan:
            return {
                "gender": loan.gender,
                "disbursement_date": loan.disbursement_date,
                "currency": loan.currency,
                "country": loan.country,
                "is_employed": loan.is_employed,
                "job": loan.job,
                "location": loan.location,
                "loan_amount": loan.loan_amount,
                "number_of_defaults": loan.number_of_defaults,
                "outstanding_balance": loan.outstanding_balance,
                "interest_rate": loan.interest_rate,
                "age": loan.age,
                "remaining_term": loan.remaining_term,
                "salary": loan.salary,
                "marital_status": loan.marital_status,
                "loan_status": loan.loan_status,
                "target": loan.target,
                "predicted_score": loan.predicted_score,
                "predicted_confidence": loan.predicted_confidence
            }
        else:
            return {"message": "Loan data not found"}

@app.put("/loan_data/{loan_id}")
def update_loan_data(loan_id: int, loan_data: LoanData):
    with engine.connect() as conn:
        conn.execute(
            LoanModel.update().where(LoanModel.id == loan_id).values(
                gender=loan_data.gender,
                disbursement_date=loan_data.disbursement_date,
                currency=loan_data.currency,
                country=loan_data.country,
                is_employed=loan_data.is_employed,
                job=loan_data.job,
                location=loan_data.location,
                loan_amount=loan_data.loan_amount,
                number_of_defaults=loan_data.number_of_defaults,
                outstanding_balance=loan_data.outstanding_balance,
                interest_rate=loan_data.interest_rate,
                age=loan_data.age,
                remaining_term=loan_data.remaining_term,
                salary=loan_data.salary,
                marital_status=loan_data.marital_status,
                loan_status=loan_data.loan_status,
                target=loan_data.target
            )
        )
        return {"message": f"Loan data for loan ID {loan_id} has been updated."}

@app.post("/classify_loans")
def classify_loans(loan_data: LoanData):
    # Prepare the data for classification
    X = pd.DataFrame([loan_data.dict()])
    y_pred = model.predict(X)
    y_conf = model.predict_proba(X)[:, 1]

    # Save the loan data to the database
    with engine.connect() as conn:
        conn.execute(
            LoanModel.insert().values(
                gender=loan_data.gender,
                disbursement_date=loan_data.disbursement_date,
                currency=loan_data.currency,
                country=loan_data.country,
                is_employed=loan_data.is_employed,
                job=loan_data.job,
                location=loan_data.location,
                loan_amount=loan_data.loan_amount,
                number_of_defaults=loan_data.number_of_defaults,
                outstanding_balance=loan_data.outstanding_balance,
                interest_rate=loan_data.interest_rate,
                age=loan_data.age,
                remaining_term=loan_data.remaining_term,
                salary=loan_data.salary,
                marital_status=loan_data.marital_status,
                loan_status=loan_data.loan_status,
                target=loan_data.target,
                predicted_score=float(y_pred[0]),
                predicted_confidence=float(y_conf[0])
            )
        )

    return {"prediction": bool(y_pred[0]), "confidence": float(y_conf[0])}

@app.get("/loans")
def list_loans(limit: int = 10, offset: int = 0):
    with engine.connect() as conn:
        loans = conn.execute(
            LoanModel.select().offset(offset).limit(limit)
        ).all()
        return [
            {
                "id": loan.id,
                "gender": loan.gender,
                "disbursement_date": loan.disbursement_date,
                "currency": loan.currency,
                "country": loan.country,
                "is_employed": loan.is_employed,
                "job": loan.job,
                "location": loan.location,
                "loan_amount": loan.loan_amount,
                "number_of_defaults": loan.number_of_defaults,
                "outstanding_balance": loan.outstanding_balance,
                "interest_rate": loan.interest_rate,
                "age": loan.age,
                "remaining_term": loan.remaining_term,
                "salary": loan.salary,
                "marital_status": loan.marital_status,
                "loan_status": loan.loan_status,
                "target": loan.target,
                "predicted_score": loan.predicted_score,
                "predicted_confidence": loan.predicted_confidence
            }
            for loan in loans
        ]

@app.get("/loans/search")
def search_loans(
    gender: str = None,
    country: str = None,
    job: str = None,
    loan_status: str = None
):
    with engine.connect() as conn:
        query = LoanModel.select()
        if gender:
            query = query.where(LoanModel.gender == gender)
        if country:
            query = query.where(LoanModel.country == country)
        if job:
            query