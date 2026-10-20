"""Register V5 only in its explicit entry points; V1/V2/V4 registrations unchanged."""
from vimeml.training.model_factory import ARCHITECTURES
from vimeml.training.model_v5 import GPTV5Config, TinyGPTV5

ARCHITECTURE = "tiny_gpt_v5"
FORMAT = "vimeml_tiny_gpt_v5"


def register_v5():
    definition = (GPTV5Config, TinyGPTV5, FORMAT)
    if ARCHITECTURE in ARCHITECTURES and ARCHITECTURES[ARCHITECTURE] != definition:
        raise ValueError("Conflicting V5 registration")
    ARCHITECTURES[ARCHITECTURE] = definition


register_v5()
