from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
CLASS_NAMES = ["no_tumor", "tumor"]
MODEL_PATH = ARTIFACTS_DIR / "brain_tumor_cnn.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
