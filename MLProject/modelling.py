"""MLflow Project entry point for the manual tuning workflow."""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# MLflow Projects supplies a parent run ID to the entry point. The training
# script creates separate manual runs for each model, so avoid resuming that
# orchestration run accidentally.
os.environ.pop("MLFLOW_RUN_ID", None)

from src.train_tuning import main


if __name__ == "__main__":
    main()
