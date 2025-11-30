# Scripts Directory

Utility scripts for working with the project.

## Available Scripts

### train_model.py
Model training and artifact saving.

```bash
cd scripts
python3 train_model.py
```

The script:
1. Loads data from `../data/raw/train.csv`
2. Performs preprocessing
3. Trains RandomForestClassifier
4. Saves model to `../models/model_artifacts.joblib`

### test_api.py
API endpoint testing.

```bash
# First, start the API server
cd ..
uvicorn app.main:app --port 8080

# In another terminal, run tests
cd scripts
python3 test_api.py
```

The script tests:
- `/health` - health check
- `/` - model information
- `/predict` - single prediction
- `/predict_batch` - batch predictions
