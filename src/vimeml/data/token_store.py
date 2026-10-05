"""Read full sentence token sequences using read-only memory maps."""

import json
import mmap
import struct
from pathlib import Path


class TokenStore:
    def __init__(self, directory, split="train"):
        self.directory = Path(directory)
        self._resources = []
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split.")
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or self.manifest.get("format") != "vimeml_sentence_tokens_v1":
            raise ValueError("Incomplete or unsupported token store.")
        if (self.manifest.get("token_dtype"), self.manifest.get("offset_dtype"), self.manifest.get("offset_unit")) != ("uint16_le", "uint64_le", "tokens"):
            raise ValueError("Unsupported token or offset format.")
        self.split = split
        stats = self.manifest["splits"][split]
        self.count = stats["sentences"]
        self.total_tokens = stats["stored_tokens"]
        try:
            self._tokens = self._map(f"{split}.tokens.bin", self.total_tokens * 2)
            self._offsets = self._map(f"{split}.offsets.bin", (self.count + 1) * 8)
            if self._offset(0) != 0 or self._offset(self.count) != self.total_tokens:
                raise ValueError("Invalid initial or final sentence offset.")
        except BaseException:
            self.close()
            raise

    def _map(self, name, expected_size):
        path = self.directory / name
        if path.stat().st_size != expected_size:
            raise ValueError(f"Unexpected file size: {name}")
        stream = path.open("rb")
        self._resources.append(stream)
        if expected_size == 0:
            return None
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._resources.append(mapped)
        return mapped

    def _offset(self, index):
        return struct.unpack_from("<Q", self._offsets, index * 8)[0]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not self._resources:
            raise ValueError("TokenStore is closed.")
        if not isinstance(index, int):
            raise TypeError("Sentence index must be an integer.")
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        start, end = self._offset(index), self._offset(index + 1)
        if not 0 <= start < end <= self.total_tokens:
            raise ValueError("Invalid sentence offset range.")
        ids = [item[0] for item in struct.iter_unpack("<H", self._tokens[start * 2:end * 2])]
        special = self.manifest["special_ids"]
        if len(ids) < 3 or ids[0] != special["bos"] or ids[-1] != special["eos"]:
            raise ValueError("Invalid sentence boundary tokens.")
        if any(token >= self.manifest["vocab_size"] for token in ids):
            raise ValueError("Token ID outside vocabulary.")
        return ids

    def close(self):
        while self._resources:
            self._resources.pop().close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
