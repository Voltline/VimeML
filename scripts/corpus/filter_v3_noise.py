"""Exclude obvious catalogue/price-list chunks from training without re-encoding."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from vimeml.training.data import file_sha, write_json

PRICE = re.compile(r"[¥￥]\s*\d[\d,，.．]*|\d[\d,，.．]*\s*円")
COMMERCE = re.compile(r"送料無料|税込|商品価格|販売価格|商品金額|在庫|サイズ\s*[:：]|カラー\s*[:：]|カート|最安値|最新作|直送品|代引不可|通販|新品|SALE|[%％]\s*OFF", re.I)


def reason(text):
    endings = len(re.findall(r"[。！？!?]", text))
    prose_sparse = endings < max(2, len(text) / 180)
    if prose_sparse and len(PRICE.findall(text)) >= 4:
        return "dense_price_list_with_sparse_sentences"
    if prose_sparse and len(COMMERCE.findall(text)) >= 6:
        return "dense_catalogue_markers_with_sparse_sentences"
    return None


def main():
    directory = ROOT / "artifacts/training-data/corpus-v3-noise-filter"
    directory.mkdir(parents=True, exist_ok=True)
    token_dir = ROOT / "artifacts/token-data/corpus-v3-half/mixed"
    tm = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
    if (directory / "manifest.json").exists():
        raise ValueError("Filter already exists; preserve it or select a new version")
    excluded = np.zeros(tm["splits"]["train"]["sentences"], dtype=np.bool_)
    counts, reasons, examples = Counter(), Counter(), []
    with (token_dir / "train.jsonl").open(encoding="utf-8") as stream:
        count = 0
        for i, line in enumerate(stream):
            row = json.loads(line)
            why = reason(row["text"]) if row["source"].startswith("jpnmix:") else None
            if why:
                excluded[i] = True
                counts[row["source"]] += 1
                reasons[why] += 1
                if len(examples) < 24:
                    examples.append({"sequence": i, "source": row["source"], "reason": why, "text": row["text"]})
            count += 1
    if count != len(excluded):
        raise ValueError("Text rows differ from token sequence count")
    np.save(directory / "train.excluded.npy", excluded, allow_pickle=False)
    write_json(directory / "manifest.json", {"status": "complete", "token_manifest_sha256": file_sha(token_dir / "manifest.json"),
        "file": "train.excluded.npy", "bytes": (directory / "train.excluded.npy").stat().st_size,
        "excluded_sequences": int(excluded.sum()), "sources": dict(counts), "reasons": dict(reasons),
        "scope": "Training web chunks only; chat unchanged; validation/test retain their original broad distribution",
        "rules": "At least 4 prices or 6 commerce markers AND sparse sentence punctuation; no Edu or topic scoring",
        "limitations": "Conservative targeted catalogue exclusion; no general text-quality certification"})
    write_json(ROOT / "outputs/training-v3/noise-exclusion-examples.json", examples)
    print(json.dumps({"excluded": int(excluded.sum()), "sources": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
