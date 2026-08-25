"""
Salary Prediction API
----------------------
Serves the trained scikit-learn Ridge regression pipeline
(preprocessing + model, saved via joblib) behind a FastAPI endpoint.

Run locally:
    uvicorn main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "salary_prediction_model.pkl"

# Categories the model was trained on — validated at the API boundary
# so bad input fails fast with a clear error instead of a silent misprediction.
VALID_EXPERIENCE = ["EN", "MI", "SE", "EX"]
VALID_EMPLOYMENT = ["FT", "PT", "CT", "FL"]
VALID_JOB_TITLES = [
    "Data Scientist", "Data Engineer", "Data Analyst", "Machine Learning Engineer",
    "ML Engineer", "Research Scientist", "Analytics Engineer", "BI Analyst",
    "Data Architect", "AI Engineer", "Applied Scientist", "Data Science Manager",
]
VALID_COMPANY_SIZE = ["S", "M", "L"]
VALID_LOCATIONS = ["US", "GB", "DE", "CA", "IN", "FR", "ES", "NG", "BR", "AU", "NL", "PT"]

model_pipeline = None  # loaded on startup


class ExperienceLevel(str, Enum):
    entry = "EN"
    mid = "MI"
    senior = "SE"
    executive = "EX"


class EmploymentType(str, Enum):
    full_time = "FT"
    part_time = "PT"
    contract = "CT"
    freelance = "FL"


class CompanySize(str, Enum):
    small = "S"
    medium = "M"
    large = "L"


class SalaryPredictionRequest(BaseModel):
    work_year: int = Field(..., ge=2020, le=2030, description="Year of employment, e.g. 2025")
    experience_level: ExperienceLevel = Field(..., description="EN=Entry, MI=Mid, SE=Senior, EX=Executive")
    employment_type: EmploymentType = Field(..., description="FT/PT/CT/FL")
    job_title: str = Field(..., description=f"One of: {', '.join(VALID_JOB_TITLES)}")
    remote_ratio: int = Field(..., ge=0, le=100, description="0, 50, or 100")
    company_size: CompanySize = Field(..., description="S/M/L")
    company_location: str = Field(..., description=f"ISO-ish country code, one of: {', '.join(VALID_LOCATIONS)}")
    employee_residence: str = Field(..., description="Country code of the employee's residence")

    model_config = {
        "json_schema_extra": {
            "example": {
                "work_year": 2025,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_title": "Data Scientist",
                "remote_ratio": 100,
                "company_size": "M",
                "company_location": "US",
                "employee_residence": "US",
            }
        }
    }


class SalaryPredictionResponse(BaseModel):
    predicted_salary_usd: float
    predicted_salary_range_usd: dict
    input_echo: SalaryPredictionRequest


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def build_features(req: SalaryPredictionRequest) -> pd.DataFrame:
    """Reproduce the exact feature engineering used at training time."""
    job_title_grouped = req.job_title if req.job_title in VALID_JOB_TITLES else "Other"
    if req.job_title not in VALID_JOB_TITLES:
        # Titles outside the known list are treated the same way rare titles
        # were grouped during training: fall back to 'Other'.
        job_title_grouped = "Other"

    row = {
        "years_since_2021": req.work_year - 2021,
        "remote_ratio": float(req.remote_ratio),
        "is_remote": 1 if req.remote_ratio == 100 else 0,
        "same_country": 1 if req.employee_residence == req.company_location else 0,
        "experience_level": req.experience_level.value,
        "employment_type": req.employment_type.value,
        "job_title_grouped": job_title_grouped,
        "company_size": req.company_size.value,
        "company_location": req.company_location,
    }
    return pd.DataFrame([row])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model_pipeline = joblib.load(MODEL_PATH)
    yield
    model_pipeline = None


app = FastAPI(
    title="Salary Prediction API",
    description="Predicts data-science salaries (USD) from job/company attributes using a tuned Ridge regression pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(status="ok", model_loaded=model_pipeline is not None)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model_pipeline is not None)


@app.post("/predict", response_model=SalaryPredictionResponse)
def predict(req: SalaryPredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if req.company_location not in VALID_LOCATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"company_location '{req.company_location}' not recognized. Valid values: {VALID_LOCATIONS}",
        )
    if req.employee_residence not in VALID_LOCATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"employee_residence '{req.employee_residence}' not recognized. Valid values: {VALID_LOCATIONS}",
        )

    try:
        X = build_features(req)
        log_pred = model_pipeline.predict(X)[0]
        pred_usd = float(np.expm1(log_pred))

        # Rough interval using the model's known test-set RMSE (~$24.3K) as a
        # simple, honest uncertainty band — not a statistically derived CI.
        rmse_estimate = 24348.0
        low = max(0.0, pred_usd - rmse_estimate)
        high = pred_usd + rmse_estimate

        return SalaryPredictionResponse(
            predicted_salary_usd=round(pred_usd, 2),
            predicted_salary_range_usd={"low": round(low, 2), "high": round(high, 2)},
            input_echo=req,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
