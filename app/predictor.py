"""
Model predictor class for loading and running predictions
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

from .preprocessing import preprocess_input_data
from .models import WorkerFeatures

logger = logging.getLogger(__name__)


class FinancialStressPredictor:
    """
    Predictor class for financial stress classification
    """

    def __init__(self, model_path: str = "models/model_artifacts.joblib"):
        """
        Initialize predictor and load model artifacts

        Args:
            model_path: Path to saved model artifacts
        """
        self.model_path = Path(model_path)
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.train_medians = None
        self.train_means = None
        self.numerical_cols_outliers = None
        self.numerical_features = None
        self.categorical_features = None
        self.feature_names = None
        self._loaded = False

        self.load_model()

    def load_model(self):
        """Load model and all artifacts from disk"""
        try:
            logger.info(f"Loading model from {self.model_path}")

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            artifacts = joblib.load(self.model_path)

            self.model = artifacts['model']
            self.preprocessor = artifacts['preprocessor']
            self.label_encoder = artifacts['label_encoder']
            self.train_medians = artifacts['train_medians']
            self.train_means = artifacts['train_means']
            self.numerical_cols_outliers = artifacts['numerical_cols_outliers']
            self.numerical_features = artifacts['numerical_features']
            self.categorical_features = artifacts['categorical_features']
            self.feature_names = artifacts['feature_names']

            self._loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data

        Args:
            data: Raw input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Remove worker_id if present (not used for prediction)
        if 'worker_id' in data.columns:
            worker_ids = data['worker_id'].copy()
            data = data.drop('worker_id', axis=1)
        else:
            worker_ids = None

        # Apply preprocessing pipeline
        processed_data = preprocess_input_data(
            data,
            self.train_medians,
            self.train_means,
            self.numerical_cols_outliers,
            self.numerical_features,
            self.categorical_features
        )

        # Ensure columns are in the correct order
        processed_data = processed_data[self.feature_names]

        return processed_data, worker_ids

    def predict_single(self, features: WorkerFeatures) -> Tuple[str, Dict[str, float]]:
        """
        Make prediction for a single worker

        Args:
            features: WorkerFeatures object

        Returns:
            Tuple of (predicted_class, probability_dict)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Convert to DataFrame
        data_dict = features.model_dump()
        df = pd.DataFrame([data_dict])

        # Store worker_id if present
        worker_id = data_dict.get('worker_id')

        # Preprocess
        processed_df, _ = self.preprocess(df)

        # Transform with preprocessor
        X_transformed = self.preprocessor.transform(processed_df)

        # Predict
        prediction = self.model.predict(X_transformed)[0]
        probabilities = self.model.predict_proba(X_transformed)[0]

        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]

        # Create probability dictionary
        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(self.label_encoder.classes_, probabilities)
        }

        return predicted_class, prob_dict

    def predict_batch(self, features_list: List[WorkerFeatures]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Make predictions for multiple workers

        Args:
            features_list: List of WorkerFeatures objects

        Returns:
            List of tuples (predicted_class, probability_dict)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Convert to DataFrame
        data_dicts = [f.model_dump() for f in features_list]
        df = pd.DataFrame(data_dicts)

        # Preprocess
        processed_df, worker_ids = self.preprocess(df)

        # Transform with preprocessor
        X_transformed = self.preprocessor.transform(processed_df)

        # Predict
        predictions = self.model.predict(X_transformed)
        probabilities = self.model.predict_proba(X_transformed)

        # Decode predictions
        predicted_classes = self.label_encoder.inverse_transform(predictions)

        # Create results
        results = []
        for pred_class, probs in zip(predicted_classes, probabilities):
            prob_dict = {
                class_name: float(prob)
                for class_name, prob in zip(self.label_encoder.classes_, probs)
            }
            results.append((pred_class, prob_dict))

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        return {
            "model_type": self.model.__class__.__name__,
            "n_estimators": getattr(self.model, 'n_estimators', None),
            "max_depth": getattr(self.model, 'max_depth', None),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "target_classes": self.label_encoder.classes_.tolist(),
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features
        }
