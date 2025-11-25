"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class StressLevel(str, Enum):
    """Financial stress level categories"""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class WorkerFeatures(BaseModel):
    """Input features for a single worker"""
    worker_id: Optional[str] = Field(None, description="Unique identifier of the worker")
    survey_month: Optional[str] = Field(None, description="Month when data was collected")
    worker_age: Optional[float] = Field(None, ge=14, le=120, description="Age of the worker")
    job_sector: Optional[str] = Field(None, description="Type of gig job")
    estimated_annual_income: Optional[float] = Field(None, ge=0, description="Self-estimated total income")
    monthly_gig_income: Optional[float] = Field(None, ge=0, description="Monthly earnings from gig work")
    num_savings_accounts: Optional[int] = Field(None, ge=0, description="Number of savings/checking accounts")
    num_credit_cards: Optional[int] = Field(None, ge=0, description="Number of active credit cards")
    avg_credit_interest: Optional[float] = Field(None, ge=0, le=100, description="Average credit card interest rate")
    num_active_loans: Optional[int] = Field(None, ge=0, description="Number of ongoing loans")
    avg_loan_delay_days: Optional[float] = Field(None, ge=0, description="Average delay in loan repayments (days)")
    missed_payment_events: Optional[int] = Field(None, ge=0, description="Number of missed/late payments")
    recent_credit_checks: Optional[int] = Field(None, ge=0, description="Credit inquiries in past 3 months")
    current_total_liability: Optional[float] = Field(None, ge=0, description="Total outstanding debt")
    credit_utilization_rate: Optional[float] = Field(None, ge=0, le=100, description="Credit used / credit limit ratio")
    credit_age_months: Optional[str] = Field(None, description="Credit history duration (e.g., '17 y. 11 m.')")
    min_payment_flag: Optional[str] = Field(None, description="Payment behavior flag: Yes/No/NM")
    monthly_investments: Optional[float] = Field(None, ge=0, description="Monthly investment amount")
    spending_behavior: Optional[str] = Field(None, description="Spending pattern category")
    end_of_month_balance: Optional[float] = Field(None, ge=0, description="Account balance at month's end")

    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": "abc123",
                "survey_month": "June",
                "worker_age": 28.0,
                "job_sector": "Writer",
                "estimated_annual_income": 72810.29,
                "monthly_gig_income": 5865.98,
                "num_savings_accounts": 4,
                "num_credit_cards": 4,
                "avg_credit_interest": 17.0,
                "num_active_loans": 3,
                "avg_loan_delay_days": 15.0,
                "missed_payment_events": 12,
                "recent_credit_checks": 3,
                "current_total_liability": 1444.26,
                "credit_utilization_rate": 32.11,
                "credit_age_months": "20 y. 7 m.",
                "min_payment_flag": "No",
                "monthly_investments": 111.89,
                "spending_behavior": "Large expenses, large payments",
                "end_of_month_balance": 557.77
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request"""
    features: WorkerFeatures


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    workers: List[WorkerFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Single prediction response"""
    worker_id: Optional[str] = Field(None, description="Worker identifier")
    predicted_stress_level: StressLevel = Field(..., description="Predicted financial stress level")
    prediction_probabilities: dict = Field(..., description="Probability for each class")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    model_version: str
    features_count: int
    target_classes: List[str]
    description: str
