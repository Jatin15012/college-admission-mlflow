"""Central configuration for the college admission project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "College_Admission.csv"

TARGET = "admission_status"
POSITIVE_CLASS = "admitted"

# Removed before training — see notebooks/01_eda.ipynb
DROP_COLS = [
    "student_id",              # identifier, no predictive meaning
    "admission_probability",   # leakage: the label's generating score
    "scholarship_eligibility", # leakage: determined after admission
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "college-admission"
MODEL_NAME = "college-admission-classifier"