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
  decoder cannot emit;
- counts are kept in sorted numpy arrays (context key, next character), saved as .npz.
"""
import copy
import hashlib
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
        rest = [s[common:] for s in sequences]
        width = max(len(r) for r in rest)
        inputs = torch.zeros((len(rest), width), dtype=torch.long)
        for row, r in enumerate(rest):
            inputs[row, :len(r)] = torch.tensor(r)
        mask = torch.ones((len(rest), common + width), dtype=torch.long, device=self.device)
        if common == 0:  # a one-token context: nothing to cache, run the rows whole
            logits = self.model(inputs.to(self.device), attention_mask=mask).logits
        else:
            cache = copy.deepcopy(self._prefix_cache(sequences[0][:common]))
            cache.batch_repeat_interleave(len(rest))
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
    BITS = 16  # bits per character id in a context key; contexts of up to 4 characters fit in uint64
    FIELDS = ("ctx", "start", "total", "words", "counts")

    def __init__(self, tables, chars, discounts):
        self.chars = chars                       # id -> char; id 0 is the unknown symbol
        self.index = {c: i for i, c in enumerate(chars) if i}
        self.tables, self.discounts = tables, discounts
        self.context_length = 10**6
        self.vocabulary = Vocabulary([None, *chars[1:]], "cpu")
        self._memo = OrderedDict()
        V = len(chars)
        t = tables[1]
        words, counts, total = t["words"], t["counts"], float(t["total"][0])
        self.unigram = np.full(V, discounts[1] * len(words) / total / V)
        self.unigram[words] += np.maximum(counts - discounts[1], 0) / total

    @classmethod
    def _key(cls, ids):
        key = 0
        for i in ids:
            key = (key << cls.BITS) | int(i)
        return np.uint64(key)

    @staticmethod
    def _unique(ctx, word):
        """Distinct (context, next character) pairs, sorted, with how often each occurs."""
        order = np.lexsort((word, ctx))
        ctx, word = ctx[order], word[order]
        del order
        new = np.ones(len(ctx), dtype=bool)
        new[1:] = (ctx[1:] != ctx[:-1]) | (word[1:] != word[:-1])
        first = np.flatnonzero(new)
        counts = np.diff(np.append(first, len(ctx)))
        return ctx[first], word[first], counts

    @staticmethod
    def _table(ctx, word, counts):
        new = np.ones(len(ctx), dtype=bool)
        new[1:] = ctx[1:] != ctx[:-1]
        first = np.flatnonzero(new)
        return {"ctx": ctx[first], "start": np.append(first, len(ctx)).astype(np.int64),
                "total": np.add.reduceat(counts, first).astype(np.float64),
                "words": word.astype(np.int32), "counts": counts.astype(np.float32)}

    @classmethod
    def build(cls, documents, log=print):
        """Counts with numpy arrays: a few dozen bytes per character of the sample at peak."""
        chars = Counter()
        for d in documents:
            chars.update(d)
        chars[SEP] += len(documents)
        vocab = [None] + sorted(c for c, n in chars.items() if n >= cls.MIN_COUNT)
        if len(vocab) >= 2**cls.BITS:
            raise ValueError("too many distinct characters for 16-bit ids")
        index = {c: i for i, c in enumerate(vocab) if i}
        lengths = np.array([len(d) + 1 for d in documents], dtype=np.int64)
        ids = np.fromiter((index.get(c, 0) for d in documents for c in SEP + d), dtype=np.uint64,
                          count=int(lengths.sum()))
        doc = np.repeat(np.arange(len(documents), dtype=np.int32), lengths)
        del chars, lengths
        n = cls.ORDER
        start = np.flatnonzero(doc[:len(doc) - n + 1] == doc[n - 1:])  # grams inside one document
        del doc
        ctx = np.zeros(len(start), dtype=np.uint64)
        for j in range(n - 1):
            ctx = (ctx << np.uint64(cls.BITS)) | ids[start + j]
        word = ids[start + n - 1].astype(np.int32)
        del ids, start
        ctx, word, counts = cls._unique(ctx, word)
        log("counted order", n, "distinct grams", len(ctx))
        tables, discounts = {}, {}
        while True:
            n1, n2 = int(np.sum(counts == 1)), int(np.sum(counts == 2))
            discounts[n] = n1 / (n1 + 2 * n2) if n1 else 0.5
            tables[n] = cls._table(ctx, word, counts)
            log("order", n, "contexts", len(tables[n]["ctx"]), "grams", len(word),
                "discount", round(discounts[n], 4))
            if n == 1:
                break
            # the next lower order uses continuation counts: distinct left extensions
            n -= 1
            ctx = ctx & np.uint64((1 << (cls.BITS * (n - 1))) - 1)
            ctx, word, counts = cls._unique(ctx, word)
        return cls(tables, vocab, discounts)

    def save(self, path):
        arrays = {"chars": np.array(self.chars[1:], dtype="<U1"),
                  "discounts": np.array([self.discounts[n] for n in range(1, self.ORDER + 1)])}
        for n, t in self.tables.items():
            arrays.update({f"{k}{n}": t[k] for k in self.FIELDS})
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            chars = [None, *data["chars"].tolist()]
            discounts = {n: float(d) for n, d in enumerate(data["discounts"], 1)}
            tables = {n: {k: data[f"{k}{n}"] for k in cls.FIELDS} for n in range(1, cls.ORDER + 1)}
        return cls(tables, chars, discounts)

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
            t = self.tables[n]
            k = self._key(key[len(key) - (n - 1):])
            i = int(np.searchsorted(t["ctx"], k))
            if i == len(t["ctx"]) or t["ctx"][i] != k:
                continue
            a, b = t["start"][i], t["start"][i + 1]
            words, counts, total = t["words"][a:b], t["counts"][a:b], t["total"][i]
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
    """Fixed sample of export-r1/train: web+review documents and LCCC dialogues by sha256 of their id.

    Only the sampled texts are turned into Python strings; the rest stays in Arrow buffers.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    train = Path(export_dir) / "train"
    files = sorted(train.glob("*.parquet"))
    web_files = [f for f in files if not f.name.startswith("chat-")]
    chat_files = [f for f in files if f.name.startswith("chat-")]
    web_total = sum(pc.sum(pq.read_table(f, columns=["han"]).column("han")).as_py() for f in web_files)
    fraction = web_chars / web_total
    documents, web_n = [], 0
    for f in web_files:
        for batch in pq.ParquetFile(f).iter_batches(columns=["id", "text"], batch_size=4096):
            rows = [r for r, i in enumerate(batch.column("id").to_pylist()) if sample_fraction(str(i), fraction)]
            if rows:
                texts = batch.column("text").take(pa.array(rows)).to_pylist()
                documents.extend(texts)
                web_n += sum(len(t) for t in texts)
    log("web documents", len(documents), "chars", web_n, "fraction", fraction)
    chat_total = sum(pc.sum(pc.utf8_length(pq.read_table(f, columns=["text"]).column("text"))).as_py()
                     for f in chat_files)
    fraction = chat_chars / chat_total
    dialogues = {}
    for f in chat_files:
        table = pq.read_table(f, columns=["dialogue", "turn", "text"])
        column = table.column("dialogue")
        chosen = [d for d in pc.unique(column).to_pylist() if sample_fraction(f"chat:{d}", fraction)]
        table = table.filter(pc.is_in(column, value_set=pa.array(chosen, type=column.type)))
        for d, t, text in zip(*(table.column(c).to_pylist() for c in ("dialogue", "turn", "text"))):
            dialogues.setdefault(d, []).append((t, text))
        del table
    chat_n = 0
    for d in sorted(dialogues, key=str):
        text = SEP.join(x for _, x in sorted(dialogues[d]))
        documents.append(text)
        chat_n += len(text)
    log("chat dialogues", len(dialogues), "chars", chat_n, "fraction", fraction)
    return documents, {"web_chars": web_n, "chat_chars": chat_n, "documents": len(documents)}
