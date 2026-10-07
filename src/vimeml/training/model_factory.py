"""Architecture dispatch shared by training and checkpoint inference."""

from vimeml.training.model import GPTConfig, TinyGPT
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2

ARCHITECTURES = {
    "tiny_gpt_v1": (GPTConfig, TinyGPT, "vimeml_tiny_gpt_v1"),
    "tiny_gpt_v2": (GPTV2Config, TinyGPTV2, "vimeml_tiny_gpt_v2"),
}


def specification(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return ARCHITECTURES[architecture]


def configuration_for(architecture, values):
    config_type, _, _ = specification(architecture)
    return config_type(**values)


def create_model(architecture, config):
    config_type, model_type, _ = specification(architecture)
    if type(config) is not config_type:
        raise ValueError("Model configuration does not match architecture.")
    return model_type(config)


def checkpoint_format(architecture):
    return specification(architecture)[2]


def model_from_checkpoint(saved):
    architecture = next(
        (
            name
            for name, (_, _, format_name) in ARCHITECTURES.items()
            if format_name == saved.get("format")
        ),
        None,
    )
    if architecture is None:
        raise ValueError("Unsupported checkpoint format.")
    if (
        saved.get("architecture", architecture) != architecture
        or saved.get("config", {}).get("architecture", architecture) != architecture
    ):
        raise ValueError("Checkpoint architecture differs from its format.")
    model = create_model(architecture, configuration_for(architecture, saved["model_config"]))
    model.load_state_dict(saved["model"])
    return model
