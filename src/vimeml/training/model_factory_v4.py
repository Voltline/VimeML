"""Register V4 only in its explicit entry points; frozen V1/V2 factory unchanged."""
from vimeml.training.model_factory import ARCHITECTURES
from vimeml.training.model_v4 import GPTV4Config, TinyGPTV4

ARCHITECTURE = "tiny_gpt_v4"
FORMAT = "vimeml_tiny_gpt_v4"


def register_v4():
    definition = (GPTV4Config, TinyGPTV4, FORMAT)
    if ARCHITECTURE in ARCHITECTURES and ARCHITECTURES[ARCHITECTURE] != definition:
        raise ValueError("Conflicting V4 registration")
    ARCHITECTURES[ARCHITECTURE] = definition


register_v4()
