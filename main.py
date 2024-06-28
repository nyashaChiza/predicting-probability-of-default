from typing import Union, List
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import date, datetime, time, timedelta

app = FastAPI()

# Create the SQLite database engine
engine = create_engine('sqlite:///loans.db')
SessionLocal = sessionmaker(bind=engine)

# Define the SQLAlchemy model
Base = declarative_base()

class LoanModel(Base):
    __tablename__ = 'loans'

    id = Column(Integer, primary_key=True, index=True)
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
    predicted_score = Column(String)
    predicted_confidence = Column(Float)

# Create the database table if it doesn't exist
Base.metadata.create_all(engine)

class LoanData(BaseModel):
    gender: str
    disbursement_date: date
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
    loan_status: Union[str, None] = None
    target: Union[bool, None] = None

class ClassificationModel:
    def __init__(self, path):
        self.path = path
        self.model = None
        #self.load_model()
    
    def load_model(self):
        # Load the pre-trained model
        with open(self.path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, loan_data: pd.DataFrame):
        data = self.clean(loan_data)
        if data['status']:
            return 'Default'#self.model.predict(data['cleaned_data'])
        else:
            return None
    
    def clean(self, data):
        # Implement actual data cleaning steps
        cleaned_data = pd.DataFrame([data])
        return {'status': True, 'cleaned_data': cleaned_data}

classifier = ClassificationModel('models/loan_classification_model.pkl')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/loan_data/{loan_id}")
def read_loan_data(loan_id: int):
    with SessionLocal() as session:
        loan = session.query(LoanModel).filter(LoanModel.id == loan_id).one_or_none()
        if loan:
            return loan
        else:
            raise HTTPException(status_code=404, detail="Loan data not found")

@app.put("/loan_data/{loan_id}")
def update_loan_data(loan_id: int, loan_data: LoanData):
    with SessionLocal() as session:
        loan = session.query(LoanModel).filter(LoanModel.id == loan_id).one_or_none()
        if not loan:
            raise HTTPException(status_code=404, detail="Loan data not found")
        
        for key, value in loan_data.dict().items():
            setattr(loan, key, value)
        
        session.commit()
        return {"message": f"Loan data for loan ID {loan_id} has been updated."}

@app.post("/classify_loans")
def classify_loans(loan_data: LoanData):
    data_dict = loan_data.dict()
    prediction = classifier.predict(data_dict)
    
    if prediction is None:
        raise HTTPException(status_code=400, detail="Invalid data for prediction")
    
    prediction_label = 'Default' if prediction[0] == 1 else 'Non-Default'
    confidence = 82.34  # Mock confidence score; replace with actual confidence calculation

    with SessionLocal() as session:
        loan = LoanModel(**data_dict)
        loan.disbursement_date = loan_data.disbursement_date.isoformat()
        loan.predicted_score = prediction_label
        loan.predicted_confidence = confidence
        session.add(loan)
        session.commit()
    
    return {"prediction": prediction_label, "confidence": confidence}

@app.get("/loans")
def list_loans(limit: int = 10, offset: int = 0):
    with SessionLocal() as session:
        loans = session.query(LoanModel).offset(offset).limit(limit).all()
        return loans

@app.get("/loans/search")
def search_loans(
    gender: str = None,
    country: str = None,
    job: str = None,
    loan_status: str = None
):
    with SessionLocal() as session:
        query = session.query(LoanModel)
        if gender:
            query = query.filter(LoanModel.gender == gender)
        if country:
            query = query.filter(LoanModel.country == country)
        if job:
            query = query.filter(LoanModel.job == job)
        if loan_status:
            query = query.filter(LoanModel.loan_status == loan_status)
        
        loans = query.all()
        return loans
