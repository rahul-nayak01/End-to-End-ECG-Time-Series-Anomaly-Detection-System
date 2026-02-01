import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

PROJECT_NAME = "ecg_anomaly_detection"

list_of_files = [

    # GitHub Actions
    ".github/workflows/ci-cd.yaml",

    # Research
    "research/ecg_anomaly_detection.ipynb",

    # Source package
    f"src/{PROJECT_NAME}/__init__.py",

    # Data
    f"src/{PROJECT_NAME}/data/__init__.py",
    f"src/{PROJECT_NAME}/data/loader.py",
    f"src/{PROJECT_NAME}/data/preprocessing.py",

    # Features
    f"src/{PROJECT_NAME}/features/__init__.py",
    f"src/{PROJECT_NAME}/features/windowing.py",

    # Models
    f"src/{PROJECT_NAME}/models/__init__.py",
    f"src/{PROJECT_NAME}/models/lstm_autoencoder.py",
    f"src/{PROJECT_NAME}/models/train.py",
    f"src/{PROJECT_NAME}/models/evaluate.py",
    f"src/{PROJECT_NAME}/models/inference.py",

    # Config
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/config.yaml",

    # Utils
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/metrics.py",

    # Application
    "app/main.py",
    "app/schemas.py",

    # Artifacts
    "artifacts/model.pth",
    "artifacts/threshold.json",

    # Root files
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "README.md",
    ".gitignore"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if not filepath.exists():
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
