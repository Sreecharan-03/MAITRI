import os
from pathlib import Path
from dotenv import load_dotenv

# Base backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "MAITRI - Multimodal Astronaut Wellbeing System"
    VERSION: str = "1.0.0"
    BASE_DIR: Path = BASE_DIR
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model paths
    FACE_MODEL_CHECKPOINT: Path = BASE_DIR / "models" / "face_detection" / "rafdb_epoch21_acc0.897_bacc0.8275.pth"
    
    # Fallback to workspace root face_detection if needed
    if not FACE_MODEL_CHECKPOINT.exists():
        FACE_MODEL_CHECKPOINT = BASE_DIR.parent / "face_detection" / "rafdb_epoch21_acc0.897_bacc0.8275.pth"

settings = Settings()
