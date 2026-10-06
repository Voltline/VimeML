"""Manual deployment CLI: no training, API calls or automatic stage chaining."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.deployment.cli import main

if __name__ == "__main__":
    main()
