"""Step-0 adapters for ime-predict-zh-v1: Hugging Face causal LMs and a character 5-gram.

Both expose what vimeml.benchmarks.predict_zh.predict needs: `vocabulary`, `context_length`,
`context(turns, prefix)` and `logprobs(sequences)`. The decoder itself is unchanged.

HFLM (Qwen2.5 base models)
- context: earlier turns and the prefix joined by "\\n" (a base model has no chat format);
  the oldest whole turns are dropped when the text does not fit in `context_length` minus
  RESERVE tokens; no BOS (Qwen2.5 has none);
- vocabulary: a token can be decoded only if its own text is entirely Chinese characters;
  characters that the byte-level BPE splits across several tokens are unreachable;
- the context's key/value cache is computed once per decode and reused by every beam step;
  only the new tokens are run, right-padded, and each row's last real position is read.

CharNgramLM
- interpolated Kneser-Ney over characters, orders 1-5, one discount per order
  D = n1 / (n1 + 2 n2); the unigram level is interpolated with a uniform distribution;
- each document or dialogue is counted separately, starting with "\\n"; dialogue turns and
  paragraphs are separated by "\\n", which is also how earlier turns join the prefix;
- characters seen fewer than 2 times in the sample map to one unknown symbol that the
  decoder cannot emit.
"""
import copy
import hashlib
import math
import pickle
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from vimeml.benchmarks.predict_zh import RESERVE, Vocabulary

SEP = "\n"


class HFLM:
    def __init__(self, model, tokenizer, device, context_length=512):
        self.model, self.tokenizer = model.to(device).eval(), tokenizer
        self.device, self.context_length = device, context_length
        size = self.model.get_output_embeddings().weight.shape[0]
        special = set(self.tokenizer.all_special_ids)
        texts = []
        for i in range(size):
            if i >= len(self.tokenizer) or i in special:
                texts.append(None)
                continue
            text = self.tokenizer.decode([i])
            texts.append(None if "�" in text else text)
        self.vocabulary = Vocabulary(texts, "cpu")
        self._cache_key, self._cache = None, None

    @classmethod
    def from_pretrained(cls, name, revision, device, dtype, context_length=512):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
        return cls(model, tokenizer, device, context_length)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def context(self, turns, prefix):
        limit = self.context_length - RESERVE
        turns = [t for t in turns if t]
        while True:
            text = SEP.join([*turns, prefix])
            ids = self.encode(text)
            if len(ids) <= limit or not turns:
                return ids[-limit:]
            turns = turns[1:]

    @torch.inference_mode()
    def _prefix_cache(self, prefix):
        key = tuple(prefix)
        if key != self._cache_key:
            out = self.model(torch.tensor([prefix], device=self.device), use_cache=True)
            self._cache_key, self._cache = key, out.past_key_values
        return self._cache

    @torch.inference_mode()
    def logprobs(self, sequences):
        shortest = min(len(s) for s in sequences)
        common = 0
        while common < shortest - 1 and all(s[common] == sequences[0][common] for s in sequences):
            common += 1
        if self._cache_key is not None and len(self._cache_key) <= common \
                and tuple(sequences[0][:len(self._cache_key)]) == self._cache_key:
            common = len(self._cache_key)
        if common == 0:
            raise ValueError("sequences share no context")
        cache = copy.deepcopy(self._prefix_cache(sequences[0][:common]))
        rest = [s[common:] for s in sequences]
        width = max(len(r) for r in rest)
        inputs = torch.zeros((len(rest), width), dtype=torch.long)
        for row, r in enumerate(rest):
            inputs[row, :len(r)] = torch.tensor(r)
        cache.batch_repeat_interleave(len(rest))
        mask = torch.ones((len(rest), common + width), dtype=torch.long, device=self.device)
        logits = self.model(inputs.to(self.device), past_key_values=cache, attention_mask=mask,
                            use_cache=True).logits
        last = torch.tensor([len(r) - 1 for r in rest], device=self.device)
        rows = logits[torch.arange(len(rest), device=self.device), last]
        return rows.float().log_softmax(-1).cpu()


# Character 5-gram ---------------------------------------------------------------------------

def sample_fraction(key, fraction):
    """Deterministic selection: the first 8 bytes of sha256(key) below fraction * 2**64."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") < fraction * 2**64


class CharNgramLM:
    ORDER = 5
    MIN_COUNT = 2

    def __init__(self, tables, chars, discounts):
        self.chars = chars                       # id -> char; id 0 is the unknown symbol
        self.index = {c: i for i, c in enumerate(chars) if i}
        self.tables, self.discounts = tables, discounts
        self.context_length = 10**6
        self.vocabulary = Vocabulary([None, *chars[1:]], "cpu")
        self._memo = OrderedDict()
        V = len(chars)
        cont, total = tables[1][()]
        self.unigram = np.full(V, discounts[1] * len(cont[0]) / total / V)
        self.unigram[cont[0]] += np.maximum(cont[1] - discounts[1], 0) / total

    @classmethod
    def build(cls, documents, log=print):
        documents = [SEP + d for d in documents]
        chars = Counter()
        for d in documents:
            chars.update(d)
        vocab = [None] + sorted(c for c, n in chars.items() if n >= cls.MIN_COUNT)
        index = {c: i for i, c in enumerate(vocab) if i}
        counts = {n: Counter() for n in range(1, cls.ORDER + 1)}
        for d in documents:
            ids = tuple(index.get(c, 0) for c in d)
            for n in range(1, cls.ORDER + 1):
                counts[n].update(ids[i:i + n] for i in range(len(ids) - n + 1))
        log("counted", {n: len(c) for n, c in counts.items()})
        # lower orders use continuation counts: distinct left extensions
        used = {cls.ORDER: counts[cls.ORDER]}
        for n in range(cls.ORDER - 1, 0, -1):
            cont = Counter()
            for gram in counts[n + 1]:
                cont[gram[1:]] += 1
            used[n] = cont
        del counts
        tables, discounts = {}, {}
        for n, grams in used.items():
            freq = Counter(grams.values())
            discounts[n] = freq[1] / (freq[1] + 2 * freq[2]) if freq[1] else 0.5
            grouped = {}
            for gram, c in grams.items():
                grouped.setdefault(gram[:-1], ([], []))
                grouped[gram[:-1]][0].append(gram[-1])
                grouped[gram[:-1]][1].append(c)
            tables[n] = {ctx: ((np.array(w, dtype=np.int32), np.array(c, dtype=np.float32)), float(sum(c)))
                         for ctx, (w, c) in grouped.items()}
            log("order", n, "contexts", len(tables[n]), "discount", round(discounts[n], 4))
        return cls(tables, vocab, discounts)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"tables": self.tables, "chars": self.chars, "discounts": self.discounts}, f, protocol=5)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["tables"], data["chars"], data["discounts"])

    def context(self, turns, prefix):
        text = SEP + SEP.join([*[t for t in turns if t], prefix])
        return [self.index.get(c, 0) for c in text[-64:]]

    def distribution(self, history):
        key = tuple(history[-(self.ORDER - 1):])
        if key in self._memo:
            self._memo.move_to_end(key)
            return self._memo[key]
        p = self.unigram
        for n in range(2, self.ORDER + 1):
            if len(key) < n - 1:
                break
            entry = self.tables[n].get(key[len(key) - (n - 1):])
            if entry is None:
                continue
            (words, counts), total = entry
            d = self.discounts[n]
            p = p * (d * len(words) / total)
            p[words] += np.maximum(counts - d, 0) / total
        out = torch.from_numpy(np.log(p).astype(np.float32))
        self._memo[key] = out
        if len(self._memo) > 200_000:
            self._memo.popitem(last=False)
        return out

    def logprobs(self, sequences):
        return torch.stack([self.distribution(s) for s in sequences])


def char_ngram_documents(export_dir, web_chars, chat_chars, log=print):
    """Fixed sample of export-r1/train: web+review documents and LCCC dialogues by sha256 of their id."""
    import pyarrow.parquet as pq
    train = Path(export_dir) / "train"
    files = sorted(train.glob("*.parquet"))
    web_files = [f for f in files if not f.name.startswith("chat-")]
    chat_files = [f for f in files if f.name.startswith("chat-")]
    web_total = sum(pq.read_table(f, columns=["han"]).column("han").to_numpy().sum() for f in web_files)
    fraction = web_chars / web_total
    documents, web_n = [], 0
    for f in web_files:
        for batch in pq.ParquetFile(f).iter_batches(columns=["id", "text"], batch_size=20000):
            for i, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist()):
                if sample_fraction(str(i), fraction):
                    documents.append(text)
                    web_n += len(text)
    log("web documents", len(documents), "chars", web_n, "fraction", fraction)
    dialogues = {}
    for f in chat_files:
        table = pq.read_table(f, columns=["dialogue", "turn", "text"])
        for d, t, text in zip(*(table.column(c).to_pylist() for c in ("dialogue", "turn", "text"))):
            dialogues.setdefault(d, []).append((t, text))
    chat_total = sum(len(x) for turns in dialogues.values() for _, x in turns)
    fraction = chat_chars / chat_total
    chat_n = 0
    for d in sorted(dialogues, key=str):
        if sample_fraction(f"chat:{d}", fraction):
            text = SEP.join(x for _, x in sorted(dialogues[d]))
            documents.append(text)
            chat_n += len(text)
    log("chat dialogues", len(documents), "chars", chat_n, "fraction", fraction)
    return documents, {"web_chars": web_n, "chat_chars": chat_n, "documents": len(documents)}
