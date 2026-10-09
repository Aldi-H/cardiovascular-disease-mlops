"""MLflow Project entry point for fetching the raw dataset from B2."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.download_dataset import main


if __name__ == "__main__":
    main()
