"""Structured conversion codec and mmap dataset. Labels are already shifted."""
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

from .readings import RESERVED

TOKENIZER_SHA = "296a2bdf4f4862be8bda848ed1cd261f0b0fd43064181fad5a513c6ce73081b5"
FORMAT = "vimeml_conversion_v1"


class ConversionCodec:
    def __init__(self, model):
        from vimeml.training.data import file_sha
        if file_sha(model) != TOKENIZER_SHA:
            raise ValueError("V4 tokenizer identity mismatch")
        self.sp = spm.SentencePieceProcessor(model_file=str(model))
        if [self.sp.piece_to_id(x) for x in ("<CTX>", "<READ>", "<OUT>")] != [4, 5, 6]:
            raise ValueError("Structural marker identity mismatch")

    def encode(self, context, reading, output):
        if not output or any(s in field for field in (context, reading, output) for s in RESERVED):
            raise ValueError("Empty target or reserved literal")
        r, out = self.sp.encode(reading), self.sp.encode(output)
        budget = 129 - (len(r) + len(out) + 5)
        if budget < 0:
            return None
        ctx, removed = self.sp.encode(context), 0
        while len(ctx) > budget:
            # Unicode scalar boundary, never slicing encoded bytes or OUT/READ.
            removed += 1
            ctx = self.sp.encode(context[removed:])
        ids = [2, 4] + ctx + [5] + r + [6] + out + [3]
        first_output = 4 + len(ctx) + len(r)
        if self.sp.unk_id() in ids:
            raise ValueError("Unexpected UNK")
        return ids, first_output, removed


def shifted(ids, first_output):
    if not (ids[0] == 2 and ids[-1] == 3 and ids[first_output - 1] == 6):
        raise ValueError("Malformed conversion sequence")
    labels = [-100] * (first_output - 1) + ids[first_output:]
    assert len(ids) - 1 == len(labels) <= 128
    return {"input_ids": ids[:-1], "labels": labels}


class ConversionDataset:
    def __init__(self, directory, split="train"):
        self.directory = Path(directory)
        self.split = split
        if split not in ("train", "validation"):
            raise ValueError("Only train/validation are allowed")
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["status"] != "complete" or self.manifest["format"] != FORMAT:
            raise ValueError("Unfrozen conversion data")
        self.count = self.manifest["splits"][split]["examples"]
        self.ids = np.memmap(self.directory / f"{split}.tokens.bin", mode="r", dtype="<u2")
        self.offsets = np.memmap(self.directory / f"{split}.offsets.bin", mode="r", dtype="<u8")
        self.first = np.memmap(self.directory / f"{split}.out-start.bin", mode="r", dtype="u1")
        self.dropout = np.memmap(self.directory / f"{split}.reading-dropout.bin", mode="r", dtype="u1")
        if (len(self.offsets) != self.count + 1 or len(self.first) != self.count
                or len(self.dropout) != self.count or self.offsets[0] != 0 or self.offsets[-1] != len(self.ids)):
            raise ValueError("Conversion storage size mismatch")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        ids = self.ids[int(self.offsets[index]):int(self.offsets[index + 1])].tolist()
        first = int(self.first[index])
        if self.dropout[index]:
            left, right = ids.index(5) + 1, first - 1
            ids = ids[:left] + ids[right:]
            first -= right - left
        return shifted(ids, first)

    def close(self):
        for name in ("ids", "offsets", "first", "dropout"):
            value = getattr(self, name, None)
            if value is not None:
                value._mmap.close()
                setattr(self, name, None)

    def __getstate__(self):
        # Windows DataLoader spawn must reopen maps, not pickle token contents.
        return {"directory": self.directory, "split": self.split}

    def __setstate__(self, state):
        self.__init__(**state)
