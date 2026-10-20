"""V5 checkpoint inference: the V4 tokenizer, prefix rules and cached scoring."""
from vimeml.training.infer_v4 import JapaneseLMV4, score_candidates_cached
from vimeml.training.model_factory_v5 import ARCHITECTURE, FORMAT

__all__ = ["JapaneseLMV5", "score_candidates_cached"]


class JapaneseLMV5(JapaneseLMV4):
    FORMAT = FORMAT
    ARCHITECTURE = ARCHITECTURE
