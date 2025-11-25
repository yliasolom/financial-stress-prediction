# Financial Stress Prediction API

ML service for predicting financial stress levels of gig economy workers based on financial and behavioral data.

## Business Problem

### Challenge
Gig economy workers (Uber drivers, freelancers, delivery workers) face **financial instability**:
- Unpredictable income
- Lack of social security
- Difficulty managing finances
- High risk of falling into debt

### Solution
Predictive model for **early detection** of workers with high financial stress, enabling:
- **Fintech platforms** - offer personalized financial products
- **Insurance companies** - assess risks
- **Labor unions** - provide timely support
- **Employment platforms** - adapt working conditions

### Target Metric
**Accuracy** - classification accuracy for stress levels (3 classes)

## Overview

REST API project for classifying financial stress levels (Low, Moderate, High) using a Random Forest model.

## Data and Analysis

### Dataset
- **Size**: 56,000 records of gig economy workers
- **Features**: 19 financial and behavioral characteristics
- **Period**: Data collected monthly (February - August)

### Target Variable Distribution
```
Moderate stress:  52.6% (29,452 workers)
Low stress:       29.3% (16,435 workers)
High stress:      18.1% (10,113 workers)
```

**Insight**: Imbalanced data - majority in moderate stress state.

### Key Features

**Financial Metrics:**
- `monthly_gig_income` - monthly income (median: ~$3,028)
- `credit_utilization_rate` - credit usage (mean: 32.3%)
- `current_total_liability` - total debt (median: $1,163)
- `missed_payment_events` - missed payments (mean: 13.4)

**Demographics:**
- `worker_age` - age (median: 33 years)
- `job_sector` - work sphere (15 categories: Lawyer, Engineer, Doctor, etc.)

**Credit History:**
- `credit_age_months` - credit history age
- `num_credit_cards` - number of credit cards
- `num_active_loans` - active loans

### EDA Insights

1. **Correlations with target:**
   - High `missed_payment_events` → high stress
   - Low `monthly_gig_income` → high stress
   - High `credit_utilization_rate` → high stress

2. **Missing data:**
   - `monthly_gig_income`: 15% missing
   - `missed_payment_events`: 11% missing
   - Filled with median (for features with outliers) or mean

3. **Outliers:**
   - Found in 11 out of 15 numerical features
   - Especially in `estimated_annual_income` and `num_savings_accounts`
   - Processed using IQR method for proper imputation

### Technologies

- **FastAPI** - web framework
- **scikit-learn** - Random Forest classifier
- **Docker** - containerization
- **Pydantic** - data validation

## Model and Metrics

### Approach

1. **Preprocessing:**
   - Convert `credit_age_months` from string to numeric
   - Fix negative values (clip to 0)
   - Fill missing values with median/mean (depending on outliers)
   - Standardize numerical features (StandardScaler)
   - One-Hot Encoding for categorical features

2. **Model:**
   - **RandomForestClassifier**
   - n_estimators: 270 trees
   - max_depth: 35
   - random_state: 2025

3. **Validation:**
   - Train/Validation split: 80/20
   - Stratified split (preserving class proportions)

### Model Results

**Metrics on validation set (20% of train):**

```
Accuracy:           76.96%
F1-score (weighted): 76.87%
```

**Performance by class:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **High** | ~0.68 | ~0.62 | ~0.65 | 2,023 |
| **Low** | ~0.71 | ~0.68 | ~0.69 | 3,287 |
| **Moderate** | ~0.82 | ~0.86 | ~0.84 | 5,890 |

**Confusion Matrix (approximate):**
```
                 Predicted
              High   Low   Moderate
Actual High   1254   189    580
       Low     267  2236    784
       Moderate 418   408   5064
```

### Results Interpretation

**Strengths:**
- **High accuracy** for Moderate class (82%) - most frequent class
- **Good balance** of precision/recall across all classes
- **Stability** - no overfitting

**Areas for improvement:**
- High class harder to detect (only 62% recall)
- Model sometimes confuses High and Moderate stress

**Business interpretation:**
- **77% accuracy** - good result for 3-class problem
- Model suitable for **initial screening**
- For critical decisions, recommend **human verification**

### Feature Importance (Top-10)

1. `missed_payment_events` - missed payments
2. `credit_utilization_rate` - credit usage
3. `monthly_gig_income` - monthly income
4. `current_total_liability` - total debt
5. `avg_loan_delay_days` - loan delays
6. `end_of_month_balance` - end of month balance
7. `recent_credit_checks` - credit checks
8. `num_active_loans` - number of loans
9. `worker_age` - worker age
10. `monthly_investments` - investments

**Conclusion:** Financial metrics (debt, payments) more important than demographics (age, profession).

## Quick Start

### 1. Train Model

First, train the model and save artifacts:

```bash
cd scripts
python3 train_model.py
```

This creates `models/model_artifacts.joblib` with all necessary components.

### 2. Local Run

#### Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### With Docker

```bash
# Build image
docker build -t financial-stress-api .

# Run container
docker run -p 8000:8000 financial-stress-api
```

API available at: http://localhost:8000

Swagger UI: http://localhost:8000/docs

## API Endpoints

### 1. Health Check

**GET** `/health`

Check service health.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 2. Model Info

**GET** `/`

Get model information.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0.0",
  "features_count": 19,
  "target_classes": ["High", "Low", "Moderate"],
  "description": "Financial stress prediction model for gig economy workers"
}
```

### 3. Single Prediction

**POST** `/predict`

Prediction for single worker.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
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
  }'
```

Response:
```json
{
  "worker_id": "abc123",
  "predicted_stress_level": "Moderate",
  "prediction_probabilities": {
    "High": 0.15,
    "Low": 0.25,
    "Moderate": 0.60
  }
}
```

### 4. Batch Prediction

**POST** `/predict_batch`

Prediction for multiple workers (up to 1000 at once).

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "workers": [
      {
        "worker_id": "worker1",
        "worker_age": 28.0,
        "monthly_gig_income": 5000.0,
        ...
      },
      {
        "worker_id": "worker2",
        "worker_age": 35.0,
        "monthly_gig_income": 3000.0,
        ...
      }
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {
      "worker_id": "worker1",
      "predicted_stress_level": "Low",
      "prediction_probabilities": {...}
    },
    {
      "worker_id": "worker2",
      "predicted_stress_level": "Moderate",
      "prediction_probabilities": {...}
    }
  ],
  "total_processed": 2
}
```

### 5. Detailed Model Info

**GET** `/model/info`

Detailed model and feature information.

```bash
curl http://localhost:8000/model/info
```

## Real-World Use Cases

### Case 1: Fintech Platform - Personalized Offers

**Scenario:** Microloan platform wants to offer flexible terms to workers with moderate stress.

**Request:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "worker_id": "driver_12345",
      "job_sector": "Driver",
      "worker_age": 34.0,
      "monthly_gig_income": 3200.0,
      "num_credit_cards": 3,
      "avg_credit_interest": 18.5,
      "missed_payment_events": 8,
      "credit_utilization_rate": 42.0,
      "current_total_liability": 2100.0,
      "end_of_month_balance": 350.0
    }
  }'
```

**Response:**
```json
{
  "predicted_stress_level": "Moderate",
  "prediction_probabilities": {
    "High": 0.25,
    "Low": 0.15,
    "Moderate": 0.60
  }
}
```

**Action:** Offer installment with reduced rate (14% instead of 18%) to reduce stress.

---

### Case 2: Labor Union - Early Intervention

**Scenario:** Freelancers union identifies high-risk members for consultations.

**Request:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "worker_id": "freelancer_789",
      "job_sector": "Writer",
      "worker_age": 27.0,
      "monthly_gig_income": 1800.0,
      "num_active_loans": 5,
      "missed_payment_events": 18,
      "credit_utilization_rate": 48.0,
      "current_total_liability": 4200.0,
      "end_of_month_balance": 80.0
    }
  }'
```

**Response:**
```json
{
  "predicted_stress_level": "High",
  "prediction_probabilities": {
    "High": 0.72,
    "Low": 0.08,
    "Moderate": 0.20
  }
}
```

**Action:** Immediate contact to offer free financial consultation.

---

### Case 3: Employment Platform - Adaptive Conditions

**Scenario:** Uber wants to offer bonuses to low-stress drivers for retention.

**Request:** (batch for 1000 drivers)
```bash
curl -X POST http://localhost:8080/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "workers": [
      {
        "worker_id": "driver_001",
        "monthly_gig_income": 5500.0,
        "credit_utilization_rate": 22.0,
        "missed_payment_events": 2
      },
      ...
    ]
  }'
```

**Result:** Identified 230 drivers with Low stress → offer loyalty program.

---

### Case 4: Insurance Company - Risk Assessment

**Scenario:** Insurance company assesses life insurance non-payment risk.

**Logic:**
- **High stress** → increased coefficient (1.3x)
- **Moderate** → standard (1.0x)
- **Low** → reduced (0.85x)

**Premium calculation example:**
```python
base_premium = 500  # base premium
stress_multipliers = {
    "High": 1.3,
    "Moderate": 1.0,
    "Low": 0.85
}

# After API request
predicted_level = "High"
final_premium = base_premium * stress_multipliers[predicted_level]
# = 500 * 1.3 = $650/month
```

---

### Case 5: Minimal Request - Key Fields Only

**Scenario:** You only have basic worker information.

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "worker_age": 30.0,
      "monthly_gig_income": 4000.0,
      "num_credit_cards": 2,
      "credit_utilization_rate": 35.0
    }
  }'
```

**Note:** Other features automatically filled with medians from train data.

## Input Data

### Optional Fields

All fields are optional, but providing more information improves prediction quality.

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| worker_id | str | Unique worker identifier |
| survey_month | str | Data collection month |
| worker_age | float | Worker age (14-120) |
| job_sector | str | Job type (Writer, Doctor, Engineer, etc.) |
| estimated_annual_income | float | Estimated annual income |
| monthly_gig_income | float | Monthly gig income |
| num_savings_accounts | int | Number of savings accounts |
| num_credit_cards | int | Number of credit cards |
| avg_credit_interest | float | Average credit interest rate |
| num_active_loans | int | Number of active loans |
| avg_loan_delay_days | float | Average loan payment delay (days) |
| missed_payment_events | int | Number of missed payments |
| recent_credit_checks | int | Credit checks in last 3 months |
| current_total_liability | float | Total debt |
| credit_utilization_rate | float | Credit limit usage (%) |
| credit_age_months | str | Credit history age (format: "17 y. 11 m.") |
| min_payment_flag | str | Minimum payment flag (Yes/No/NM) |
| monthly_investments | float | Monthly investments |
| spending_behavior | str | Spending pattern |
| end_of_month_balance | float | End of month balance |

## Output Data

### Prediction Classes

- **Low** - low financial stress level
- **Moderate** - moderate financial stress level
- **High** - high financial stress level

### Probabilities

Each prediction returns probabilities for all classes (sum = 1.0).

## Deployment

### Local Server

```bash
# Build and run with Docker
docker build -t financial-stress-api .
docker run -d -p 8000:8000 --name stress-api financial-stress-api

# Check
curl http://localhost:8000/health
```

### Cloud Platforms

#### Heroku

```bash
# Install Heroku CLI and run
heroku login
heroku create your-app-name
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/financial-stress-api
gcloud run deploy financial-stress-api \
  --image gcr.io/PROJECT_ID/financial-stress-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS Elastic Container Service (ECS)

```bash
# Configure AWS CLI and run
aws ecr create-repository --repository-name financial-stress-api
docker tag financial-stress-api:latest YOUR_ECR_URI:latest
docker push YOUR_ECR_URI:latest
# Create ECS task and service via AWS Console
```

## Project Structure

```
.
├── app/                     # Main application
│   ├── __init__.py          # Module initialization
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic validation models
│   ├── predictor.py         # Prediction class
│   └── preprocessing.py     # Preprocessing functions
├── data/                    # Data (not in git)
│   ├── raw/                 # Raw data
│   │   ├── train.csv        # Training data
│   │   ├── test.csv         # Test data
│   │   └── *.csv            # Other data
│   └── processed/           # Processed data
├── models/                  # Trained models (not in git)
│   └── model_artifacts.joblib  # Saved model
├── notebooks/               # Jupyter notebooks
│   └── Offline Interview.ipynb  # EDA
├── scripts/                 # Utility scripts
│   ├── train_model.py       # Model training script
│   └── test_api.py          # API testing
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore            # Docker exclusions
├── .gitignore               # Git ignored files
└── README.md                # Documentation
```

## Testing

### Python Test

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": {
        "worker_age": 30.0,
        "monthly_gig_income": 4000.0,
        "num_credit_cards": 2,
        "credit_utilization_rate": 35.0,
        # ... other fields
    }
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript Test

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    features: {
      worker_age: 30.0,
      monthly_gig_income: 4000.0,
      // ... other fields
    }
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Model Performance

Validation set metrics (20% of train):
- **Accuracy**: ~0.74-0.76
- **F1-score (weighted)**: ~0.74-0.76

Model parameters:
- Algorithm: RandomForestClassifier
- n_estimators: 270
- max_depth: 35
- random_state: 2025

## Monitoring

Recommended monitoring setup:

1. **Health checks** - regular `/health` requests
2. **Logging** - all requests logged with timestamp
3. **Metrics** - request count, response time, errors

## Project Journey

### Development Stages

#### 1. Data Analysis (EDA)
- Dataset exploration: 56,000 records, 19 features
- Missing values detection (up to 15% in some columns)
- Distribution and correlation analysis
- Outlier detection using IQR method
- Dependency visualization (confusion matrix, feature importance)

**File:** `notebooks/Offline Interview.ipynb`

#### 2. Data Preprocessing
- String feature conversion (`credit_age_months`)
- Anomalous value correction (negative accounts)
- Missing value imputation:
  - Median for features with outliers
  - Mean for others
- Numerical feature standardization
- Categorical One-Hot Encoding

**File:** `app/preprocessing.py`

#### 3. Model Training
- Algorithm selection: **RandomForestClassifier**
- Hyperparameter tuning:
  - n_estimators search (10-300)
  - max_depth tuning (1-35)
  - Final: n_estimators=270, max_depth=35
- Validation: Stratified Split 80/20
- **Result: Accuracy 76.96%, F1 76.87%**

**File:** `scripts/train_model.py`

#### 4. REST API Development
- FastAPI application with endpoints:
  - `/health` - health check
  - `/` - model information
  - `/predict` - single prediction
  - `/predict_batch` - batch predictions
- Pydantic input validation
- Error handling and logging
- Auto-documentation (Swagger UI)

**Files:** `app/main.py`, `app/models.py`, `app/predictor.py`

#### 5. Containerization
- Multi-stage Dockerfile for size optimization
- .dockerignore for excluding unnecessary files
- Container health check
- Non-root user for security

**File:** `Dockerfile`

#### 6. Testing and Documentation
- Automated API tests (`scripts/test_api.py`)
- Real-world use case examples
- Complete README documentation
- Deployment instructions (Heroku, GCP, AWS)

#### 7. Project Organization
- Structured directories (data/, notebooks/, scripts/, app/)
- .gitignore for large file exclusion
- requirements.txt with dependencies
- README files in subdirectories

### Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **RandomForest** | Works well with categorical and numerical features, robust to outliers |
| **Median imputation** | More reliable than mean for features with outliers |
| **Stratified Split** | Preserves class proportions with imbalanced data |
| **FastAPI** | Fast, modern, with auto-documentation |
| **Docker** | Simplifies deployment and ensures reproducibility |
| **Pydantic** | Automatic validation and type coercion |

### Performance Metrics

- **Training time:** ~3-4 minutes (270 trees)
- **Model size:** ~500 MB (joblib)
- **Prediction time:** <100ms per request
- **Batch prediction:** ~1-2 sec for 1000 records

### Potential Improvements

1. **Class balancing** - use SMOTE to improve recall on High class
2. **Feature Engineering** - create new features (debt-to-income ratio)
3. **Model ensemble** - combine RF with XGBoost
4. **A/B testing** - built-in experimentation mechanism
5. **Monitoring** - add Prometheus metrics
6. **Retraining pipeline** - automatic retraining on new data

## License

Project created for educational purposes.

## Contact

For questions and suggestions, create an issue in the project repository.
