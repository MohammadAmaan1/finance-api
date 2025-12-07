from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import os


# Simple encoders (training ke hisaab se adjust kar sakte ho)
GENDER_MAP = {"Male": 1, "Female": 0}
MARRIED_MAP = {"Yes": 1, "No": 0}


app = FastAPI()

# CORS (frontend ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODEL LOAD (correct path) ----------
# BASE_DIR = folder jahan main.py pada hai
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")


print("Loading model from:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
    
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running fine",
        "model_loaded": True
    }

MODEL_FEATURES = [
    "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value",
    "bank_asset_value", "total_income",
    "loan_to_income_ratio", "total_assets"
]
@app.get("/info")
def model_info():
    return {
        "project": "Loan Approval Prediction",
        "developer": "Mohammad Amaan",
        "model_features": MODEL_FEATURES,
        "total_features": len(MODEL_FEATURES),
        "message": "Model metadata fetched successfully"
    }



class LoanInput(BaseModel):
    no_of_dependents: int = Field(..., ge=0)
    education: int = Field(..., ge=0)
    self_employed: int = Field(..., ge=0)
    income_annum: float  # yahan se gt=0 hata do
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float = Field(..., ge=0)
    commercial_assets_value: float = Field(..., ge=0)
    luxury_assets_value: float = Field(..., ge=0)
    bank_asset_value: float = Field(..., ge=0)
    total_income: float  # yahan se bhi gt=0 hata do
    loan_to_income_ratio: float
    total_assets: float = Field(..., ge=0)



@app.get("/")
def home():
    return {"status": "API running", "info": "Loan Approval Prediction"}


@app.post("/predict")
def predict_loan(data: LoanInput):

    # Basic business checks (optional)
    if data.income_annum <= 0 or data.total_income <= 0:
        raise HTTPException(
            status_code=400,
            detail="Income values must be positive."
        )

    if data.loan_amount <= 0 or data.loan_term <= 0:
        raise HTTPException(
            status_code=400,
            detail="Loan amount and term must be positive."
        )

    try:
        row = {
            "no_of_dependents": data.no_of_dependents,
            "education": data.education,
            "self_employed": data.self_employed,
            "income_annum": data.income_annum,
            "loan_amount": data.loan_amount,
            "loan_term": data.loan_term,
            "cibil_score": data.cibil_score,
            "residential_assets_value": data.residential_assets_value,
            "commercial_assets_value": data.commercial_assets_value,
            "luxury_assets_value": data.luxury_assets_value,
            "bank_asset_value": data.bank_asset_value,
            "total_income": data.total_income,
            "loan_to_income_ratio": data.loan_to_income_ratio,
            "total_assets": data.total_assets,
        }

        df = pd.DataFrame([row])

        prediction = model.predict(df)[0]
        proba = float(max(model.predict_proba(df)[0]))
        status = "Approved" if prediction == 1 else "Rejected"

        return {
            "status": "success",
            "loan_status": status,
            "probability": proba,
            "message": "Prediction generated successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while predicting: {str(e)}"
        )
