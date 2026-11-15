"""Single-node data parallelism for the zh pretraining trainers (launched with torchrun).

Every rank draws the same global batch from the same seeded loaders, then trains on its
own contiguous share of the rows. The data order, the cursor and the per-update token
normalization therefore stay exactly those of a one-device run: each rank scales its loss
by the global supervised-token count and by the world size (DDP averages gradients), so
the summed gradient is the one-device gradient up to floating-point order. Rank 0 writes
checkpoints, logs and validation; the other ranks wait for it.

With WORLD_SIZE unset or 1 nothing here changes the one-device path.
"""
import datetime
import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from vimeml.posttraining.streams import slice_batch

# Rank 0 validates alone at milestones while the others wait at a barrier; a full
# FP32 validation of a large model can take far longer than NCCL's default 10 minutes.
TIMEOUT = datetime.timedelta(hours=6)


class Context:
    def __init__(self, world=1, rank=0, local_rank=0):
        self.world, self.rank, self.local_rank = world, rank, local_rank

    @property
    def active(self):
        return self.world > 1

    @property
    def main(self):
        return self.rank == 0

    def device(self, kind):
        if kind == "cuda":
            return torch.device("cuda", self.local_rank)
        return torch.device(kind)

    def wrap(self, model, device):
        """The module that runs training forwards; `model` itself stays the unwrapped one."""
        if not self.active:
            return model
        from torch.nn.parallel import DistributedDataParallel
        ids = [device.index] if device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=ids, broadcast_buffers=False)

    def no_sync(self, module, last):
        """Skip the gradient all-reduce on every backward but the last of an update."""
        return nullcontext() if last or not self.active else module.no_sync()

    def share(self, batch):
        """This rank's contiguous rows of a global batch, trimmed to its own width
        (as posttraining.streams.microbatches trims). None when this rank gets no rows."""
        rows = len(batch["lengths"])
        start, end = rows * self.rank // self.world, rows * (self.rank + 1) // self.world
        if start == end:
            return None
        part = slice_batch(batch, start, end)
        width = math.ceil(int(part["lengths"].max()) / 8) * 8
        part["input_ids"] = part["input_ids"][:, :width]
        part["labels"] = part["labels"][:, :width]
        return part

    def sum(self, values, device):
        """Elementwise sum of a list of floats over all ranks (FP64)."""
        if not self.active:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(tensor)
        return tensor.tolist()

    def any(self, flag, device):
        """True on every rank when any rank's flag is set (stop requests)."""
        if not self.active:
            return bool(flag)
        return self.sum([1.0 if flag else 0.0], device)[0] > 0

    def broadcast(self, value):
        """Rank 0's picklable value on every rank (milestone reports)."""
        if not self.active:
            return value
        holder = [value]
        dist.broadcast_object_list(holder, src=0)
        return holder[0]

    def barrier(self):
        if self.active:
            dist.barrier()

    def close(self):
        if self.active and dist.is_initialized():
            dist.destroy_process_group()


class NullWriter:
    """Stands in for SummaryWriter on ranks other than 0."""
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def setup(device_kind):
    """Read torchrun's environment; initialize the process group when WORLD_SIZE > 1."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return Context()
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if device_kind == "cuda":
        if torch.cuda.device_count() <= local:
            raise ValueError("Data parallelism needs one visible GPU per local rank")
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if device_kind == "cuda" else "gloo", timeout=TIMEOUT)
    return Context(world, rank, local)
