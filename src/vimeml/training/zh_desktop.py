"""Model classes of the zh desktop checkpoints (zh-200m, zh-800m), looked up from a checkpoint's header."""
import importlib

DESKTOP = {("vimeml_tiny_gpt_zh200m", "tiny_gpt_zh200m"): ("model_zh200m", "GPTZh200mConfig", "TinyGPTZh200m"),
           ("vimeml_tiny_gpt_zh800m", "tiny_gpt_zh800m"): ("model_zh800m", "GPTZh800mConfig", "TinyGPTZh800m")}


def desktop_family(saved):
    """(config class, model class) of a zh-200m / zh-800m checkpoint, or None for any other checkpoint."""
    entry = DESKTOP.get((saved.get("format"), saved.get("architecture")))
    if entry is None:
        return None
    module = importlib.import_module("vimeml.training." + entry[0])
    return getattr(module, entry[1]), getattr(module, entry[2])
