"""Prepare full V4 train/validation tokens and context-128 windows."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.tokenizer.encode_v4 import main

if __name__ == "__main__":
    main()
