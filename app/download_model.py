"""
Download model artifacts from cloud storage if not present locally
"""
import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "model_artifacts.joblib"
# Set this via environment variable MODEL_URL
MODEL_URL = os.getenv("MODEL_URL", "")

def download_model():
    """Download model if it doesn't exist"""
    if MODEL_PATH.exists():
        logger.info(f"Model already exists at {MODEL_PATH}")
        return True

    if not MODEL_URL:
        logger.warning("MODEL_URL environment variable not set. Skipping download.")
        return False

    logger.info(f"Downloading model from {MODEL_URL}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Support for Dropbox links (add dl=1 parameter)
        url = MODEL_URL
        if "dropbox.com" in url and "dl=0" in url:
            url = url.replace("dl=0", "dl=1")
        elif "dropbox.com" in url and "dl=" not in url:
            url = url + ("&" if "?" in url else "?") + "dl=1"

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 50) == 0:  # Log every 50MB
                            logger.info(f"Downloaded {progress:.1f}%")

        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()  # Remove partial download
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_model()
