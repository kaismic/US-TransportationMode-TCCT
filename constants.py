from pathlib import Path
from typing import Final

ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / "data"
RAW_DATA_FILE_NAME = "raw_data.tar.gz"
RAW_DATA_EXTRACTED_PATH = DATA_PATH / "raw_data"
CLEANED_DATA_DIR = "cleaned_data"
TRANSFORMED_DATA_DIR = "transformed_data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
RUNS_DIR = "runs"

class RawDataFieldLocation:
    TIME: Final = 0
    SENSOR_TYPE: Final = 1
