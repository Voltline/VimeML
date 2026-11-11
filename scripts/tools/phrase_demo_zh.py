"""Launch the local vimeml-tiny-zh-v1 prediction page."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tools.phrase_demo_zh import main  # noqa: E402

if __name__ == "__main__":
    main()
