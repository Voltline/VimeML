"""Launch the local short-phrase UI or run its fixed qualitative suite."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tools.phrase_demo import main

if __name__ == "__main__":
    main()
