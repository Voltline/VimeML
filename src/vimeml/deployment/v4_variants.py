"""Frozen V4-family deployment variants: one checkpoint, fixture and CUDA reference each.

All variants share the V4 contract (8K tokenizer, d384/L7/6 heads, SwiGLU 960,
context 128, tied head). Paths are relative to the repository root.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    step: int
    epoch: int
    checkpoint: str
    reference: str
    cuda_scores: str
    cuda_label: str
    package_prefix: str
    resource_tag: str

    def package(self, kind):
        """Deployment directory name, e.g. `int8-b32` -> tiny-ja-v4.3-epoch30-int8-b32-v1."""
        return f"{self.package_prefix}-{kind}-v1"


def _v43(epoch, step):
    return Variant(
        name=f"v43-epoch{epoch}", label=f"V4.3 epoch {epoch}", step=step, epoch=epoch,
        checkpoint=f"artifacts/models/tiny-ja-v43-e8k-d384-l7/epoch{epoch}-deployment.pt",
        reference=f"outputs/deployment/tiny-ja-v43-epoch{epoch}-fp32-input",
        cuda_scores=f"outputs/evaluation-v43-20261020/epoch{epoch}",
        cuda_label=f"V4.3 epoch{epoch} FP32 CUDA (2026-10-20)",
        package_prefix=f"tiny-ja-v4.3-epoch{epoch}", resource_tag=f"V43E{epoch}")


VARIANTS = {variant.name: variant for variant in (
    Variant(name="v42-epoch20", label="V4.2 epoch 20", step=1081942, epoch=20,
            checkpoint="artifacts/models/tiny-ja-v42-e8k-d384-l7/deployment.pt",
            reference="outputs/deployment/tiny-ja-v42-epoch20-fp32-input",
            cuda_scores="outputs/evaluation-v42-20261020",
            cuda_label="V4.2 FP32 CUDA (2026-10-20)",
            package_prefix="tiny-ja-v4.2-epoch20", resource_tag="V42"),
    _v43(30, 3225888),
    _v43(27, 2903298),
)}
DEFAULT = "v42-epoch20"
