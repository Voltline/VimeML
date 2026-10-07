"""Post-hoc conservative spelling sensitivity; never replace frozen gold."""

import functools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.comparison import paired, rows
from vimeml.benchmarks.evaluate_ajimee import metrics
from vimeml.training.data import write_json

KANJI = re.compile("[一-龠々]")


def main():
    import fugashi
    from importlib.metadata import version

    tagger = fugashi.Tagger()

    @functools.lru_cache(maxsize=None)
    def tokens(text):
        result = []
        words = tagger(text)
        if "".join(word.surface for word in words) != text:
            return ((("unpreserved_text", text), text, "".join(KANJI.findall(text))),)
        for word in words:
            f = word.feature
            # Preserve exact lexical identity, reading and inflection;
            # unknowns remain exact. Never collapse all words to kana.
            key = (
                ("unknown", word.surface)
                if word.is_unk
                else (f.lemma, f.lForm, f.kana, f.pos1, f.pos2, f.pos3, f.cType, f.cForm)
            )
            result.append((key, word.surface, "".join(KANJI.findall(word.surface))))
        return tuple(result)

    def equivalent(a, b):
        left, right = tokens(a), tokens(b)
        if len(left) != len(right):
            return False
        for (ka, sa, ca), (kb, sb, cb) in zip(left, right):
            if ka != kb or ca and cb and ca != cb:
                return False
        return True

    output = ROOT / "outputs/ime-eval/expanded-v21-dev-draft-label-sensitivity-v2"
    if output.exists():
        raise ValueError("Preserve prior sensitivity analysis.")
    output.mkdir(parents=True)
    before = rows(ROOT / "outputs/ime-eval/expanded-v21-dev-draft-v1/scores.jsonl")
    after = rows(ROOT / "outputs/ime-eval/expanded-v21-dev-draft-v2/scores.jsonl")
    assert len(before) == len(after) == 2000
    alias_by_id = {}
    additions = []
    for a, b in zip(before, after):
        assert (
            a["id"] == b["id"]
            and a["answers"] == b["answers"]
            and a["orders"]["azookey"] == b["orders"]["azookey"]
        )
        aliases = [
            text
            for text in a["orders"]["azookey"]
            if text not in a["answers"] and any(equivalent(text, gold) for gold in a["answers"])
        ]
        alias_by_id[a["id"]] = a["answers"] + aliases
        if aliases:
            additions.append(
                {
                    "id": a["id"],
                    "context": a["left_context"],
                    "reading": a["query"],
                    "original_answers": a["answers"],
                    "diagnostic_aliases": aliases,
                    "human_verified": False,
                }
            )
    # Copy only the row dictionaries; all pool, order and score data is reused.
    diagnostic_a = [{**r, "answers": alias_by_id[r["id"]]} for r in before]
    diagnostic_b = [{**r, "answers": alias_by_id[r["id"]]} for r in after]
    result = paired(diagnostic_a, diagnostic_b)
    result.update(
        {
            "format": "expanded_ime_posthoc_spelling_sensitivity_v1",
            "posthoc": True,
            "labels_formal_gold": False,
            "not_a_replacement_label_version": True,
            "blind_lm_scored": False,
            "cases_with_proposed_aliases": len(additions),
            "proposed_aliases": sum(len(r["diagnostic_aliases"]) for r in additions),
            "azookey": metrics(diagnostic_a, "azookey"),
            "dictionary_versions": {n: version(n) for n in ("fugashi", "unidic-lite")},
            "rule": "Same UniDic lemma/lForm/kana/POS/inflection token sequence; OOV and unpreserved whitespace exact; if both surfaces contain kanji require the same kanji string. Punctuation and token boundaries retained.",
            "scope": "Apply the same fixed lexical rule to every development reference and every real candidate, irrespective of model order; no source/pool/order changes.",
            "limitations": [
                "Post-hoc dictionary proxy, not independently adjudicated labels",
                "May miss legitimate aliases or accept inappropriate spellings",
                "Primary strict-reference scores remain unchanged",
            ],
        }
    )
    write_json(output / "sensitivity.json", result)
    write_json(output / "proposed-aliases.json", additions)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "cases_with_proposed_aliases",
                    "proposed_aliases",
                    "azookey",
                    "v1",
                    "v2",
                    "paired",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
