import os
from pathlib import Path

DATASET_ROOT = r"C:\Users\indra\.cache\kagglehub\datasets\puneet6060\intel-image-classification\versions\2"

TRAIN_PATH = os.path.join(DATASET_ROOT, "seg_train", "seg_train")
TEST_PATH = os.path.join(DATASET_ROOT, "seg_test", "seg_test")
PRED_PATH = os.path.join(DATASET_ROOT, "seg_pred", "seg_pred")

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
