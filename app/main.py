"""
FastAPI application for financial stress prediction
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from .predictor import FinancialStressPredictor
from .download_model import download_model
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    global predictor
    logger.info("Starting up application...")
    try:
        # Download model if not present
        logger.info("Checking for model artifacts...")
        download_model()

        # Load predictor
        predictor = FinancialStressPredictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Financial Stress Prediction API",
    description="API for predicting financial stress levels of gig economy workers",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=ModelInfoResponse)
async def root():
    """
    Get API and model information
    """
    try:
        model_info = predictor.get_model_info()

        return ModelInfoResponse(
            model_type=model_info["model_type"],
            model_version=__version__,
            features_count=model_info["n_features"],
            target_classes=model_info["target_classes"],
            description="Financial stress prediction model for gig economy workers"
        )
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded() else "unhealthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        version=__version__
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction

    Args:
        request: PredictionRequest with worker features

    Returns:
        PredictionResponse with predicted stress level and probabilities
    """
    try:
        if not predictor or not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Make prediction
        predicted_class, probabilities = predictor.predict_single(request.features)

        # Get worker_id if provided
        worker_id = request.features.worker_id

        return PredictionResponse(
            worker_id=worker_id,
            predicted_stress_level=predicted_class,
            prediction_probabilities=probabilities
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple workers

    Args:
        request: BatchPredictionRequest with list of worker features

    Returns:
        BatchPredictionResponse with predictions for all workers
    """
    try:
        if not predictor or not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Make predictions
        results = predictor.predict_batch(request.workers)

        # Create response
        predictions = []
        for i, (predicted_class, probabilities) in enumerate(results):
            worker_id = request.workers[i].worker_id
            predictions.append(
                PredictionResponse(
                    worker_id=worker_id,
                    predicted_stress_level=predicted_class,
                    prediction_probabilities=probabilities
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_batch endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Get detailed model information
    """
    try:
        if not predictor or not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        return predictor.get_model_info()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model_info endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
