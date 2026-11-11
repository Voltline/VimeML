"""Judge the vimeml-tiny-zh-v1 arms by the rules fixed in docs/zh-CN/training-zh-v1-plan.md.

Inputs per arm, under RUNS/<arm>/:
    full.jsonl, full.meta.json        score_zh13_lm.py output on the full-pinyin track (required)
    abbrev.jsonl, abbrev.meta.json    the same on the abbreviated track (optional, descriptive)
and the arm's training directory (output_dir in configs/train-zh13-<arm>.toml, or MODELS/<arm>)
holding final.pt, summary.json and milestones.json.

Checks before any statistic: every arm is present; each scores file matches the SHA256 in its meta;
the meta names the arm, the development split and the right track; it was scored from that arm's
final.pt (SHA256 compared with the file on disk); training finished with status "complete" and all
20 milestones; arms of one vocabulary share one tokenizer and the two vocabularies differ.

Each scores file is ranked with score_standard_ime_zh.py score (the same code as every other zh
result), into OUT/scores/<arm>-<track>/; the Rime pool order is scored too, as the baseline.

Decision (full track, exact Top-1, development, 1,456 items):
    noise band  = the three 12k-fresh seeds: mean, range (max - min), SD (n - 1)
    contrasts   = the five in CONTRASTS; one family, Holm step-down on exact McNemar p, alpha 0.05
    effective   = |Top-1 difference| > seed range  AND  Holm-adjusted p < 0.05
Everything else (ep5 contrasts, BPC, memorization gap, abbreviated track, Top-3, pool-conditional
Top-1, sources, learning curves) is description and never enters the decision.
"""
import hashlib
import json
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

from vimeml.benchmarks.statistics import paired

VERSION = "ime-standard-zh-v1-r1"
ITEMS = 1456
RIME_BASELINE_TOP1 = 1001
ALPHA = 0.05
MILESTONES = 20
ARMS = {  # arm: (vocabulary, web pool, web epochs) as in the plan's arm table
    "12k-fresh-s1": (12288, "P1", 1), "12k-fresh-s2": (12288, "P1", 1), "12k-fresh-s3": (12288, "P1", 1),
    "12k-ep5": (12288, "P5", 5), "12k-ep20": (12288, "P20", 20),
    "8k-fresh": (8192, "P1", 1), "8k-ep5": (8192, "P5", 5), "8k-ep20": (8192, "P20", 20),
}
SEEDS = ("12k-fresh-s1", "12k-fresh-s2", "12k-fresh-s3")
CONTRASTS = (  # (name, A, B, question); difference is B - A
    ("data-12k", "12k-fresh-s1", "12k-ep20", "12K 下新数据与 20 轮"),
    ("data-8k", "8k-fresh", "8k-ep20", "8K 下新数据与 20 轮"),
    ("vocab-fresh", "8k-fresh", "12k-fresh-s1", "新数据下的词表"),
    ("vocab-ep5", "8k-ep5", "12k-ep5", "5 轮下的词表"),
    ("vocab-ep20", "8k-ep20", "12k-ep20", "20 轮下的词表"),
)
DESCRIPTIVE = (  # dose-response, reported only
    ("12k-fresh-s1", "12k-ep5"), ("12k-ep5", "12k-ep20"), ("8k-fresh", "8k-ep5"), ("8k-ep5", "8k-ep20"),
)
VALIDATIONS = ("web", "chat", "probe")
SCORER = Path(__file__).resolve().parents[3] / "scripts" / "benchmarks" / "score_standard_ime_zh.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def holm(pvalues):
    """Holm step-down adjusted p-values, same order as the input."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def spread(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return {"values": values, "mean": statistics.mean(values), "range": max(values) - min(values),
            "sd": statistics.stdev(values)}


def model_dir(root, arm, models):
    if models:
        return Path(models) / arm
    config = tomllib.loads((root / "configs" / f"train-zh13-{arm}.toml").read_text(encoding="utf-8"))
    if config["arm"] != arm or config["model"]["vocab_size"] != ARMS[arm][0]:
        raise ValueError(f"configs/train-zh13-{arm}.toml does not describe {arm}")
    return root / config["output_dir"]


def check_arm(arm, runs, directory, track, problems):
    """Return (scores path, meta) after checking the scores against the meta and final.pt."""
    scores, meta_path = runs / arm / f"{track}.jsonl", runs / arm / f"{track}.meta.json"
    if not scores.exists() or not meta_path.exists():
        problems.append(f"{arm}: {track}.jsonl or {track}.meta.json is missing")
        return None, None
    meta = read_json(meta_path)
    expected = {"arm": arm, "version": VERSION, "track": track, "split": "development"}
    for key, value in expected.items():
        if meta.get(key) != value:
            problems.append(f"{arm}/{track}: meta {key} is {meta.get(key)!r}, expected {value!r}")
    if sha256(scores) != meta.get("scores_sha256"):
        problems.append(f"{arm}/{track}: scores file differs from scores_sha256 in its meta")
    if Path(str(meta.get("checkpoint", "")).replace("\\", "/")).name != "final.pt":
        problems.append(f"{arm}/{track}: scored from {meta.get('checkpoint')}, not final.pt")
    final = directory / "final.pt"
    if not final.exists():
        problems.append(f"{arm}: {final} is missing")
    elif sha256(final) != meta.get("checkpoint_sha256"):
        problems.append(f"{arm}/{track}: checkpoint_sha256 differs from {final}")
    return scores, meta


def training_record(arm, directory, problems, smoke):
    summary_path, milestones_path = directory / "summary.json", directory / "milestones.json"
    if not summary_path.exists() or not milestones_path.exists():
        problems.append(f"{arm}: summary.json or milestones.json is missing in {directory}")
        return None
    summary, milestones = read_json(summary_path), read_json(milestones_path)
    allowed = ("complete", "smoke_complete") if smoke else ("complete",)
    if summary.get("status") not in allowed:
        problems.append(f"{arm}: training status is {summary.get('status')!r}")
    if len(milestones) != MILESTONES or [m["milestone"] for m in milestones] != list(range(1, MILESTONES + 1)):
        problems.append(f"{arm}: milestones.json does not hold milestones 1-{MILESTONES}")
    elif milestones[-1]["step"] != summary.get("step"):
        problems.append(f"{arm}: milestone {MILESTONES} is at step {milestones[-1]['step']}, final at {summary.get('step')}")
    return {"status": summary.get("status"), "step": summary.get("step"), "milestones": milestones}


def run_scorer(frozen, out, track, scores=None, name=None):
    command = [sys.executable, "-I", "-X", "utf8", str(SCORER), "score", str(frozen), str(out), "--track", track]
    if scores:
        command += ["--scores", str(scores)]
    if name:
        command += ["--name", name]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if done.returncode:
        raise RuntimeError(f"score_standard_ime_zh.py failed for {out.name}:\n{done.stderr}")
    return read_json(out / "summary.json"), {r["id"]: r for r in read_jsonl(out / "items.jsonl")}


def hits(rows, ids, k=1):
    return [rows[i]["rank"] is not None and rows[i]["rank"] <= k for i in ids]


def compare(rows_a, rows_b, ids):
    labels = [rows_a[i]["dialogue"] for i in ids]  # one item per dialogue in this benchmark
    return {f"top{k}": paired(hits(rows_a, ids, k), hits(rows_b, ids, k), labels) for k in (1, 3)}


def winner(entry):
    if not entry["effective"]:
        return None
    return entry["b"] if entry["difference"] > 0 else entry["a"]


def decide(contrasts):
    """Apply the plan's "结论如何使用" rules.

    Data: no effective contrast -> fresh; effective ones agree -> that; 12K and 8K effective in
    opposite directions -> the 12K result. Vocabulary: none effective -> 12K; agree -> that;
    disagree -> the fresh-data contrast if effective, otherwise 12K.
    """
    by_name = {c["name"]: c for c in contrasts}
    data_votes = {}
    for name in ("data-12k", "data-8k"):
        w = winner(by_name[name])
        if w is not None:
            data_votes[name] = "repeat-20" if w.endswith("ep20") else "fresh"
    if not data_votes:
        data = {"choice": "fresh", "basis": "两个数据对比都不有效，按计划默认新数据"}
    elif len(set(data_votes.values())) == 1:
        data = {"choice": next(iter(data_votes.values())), "basis": f"有效对比：{', '.join(sorted(data_votes))}"}
    else:
        data = {"choice": data_votes["data-12k"], "basis": "12K 与 8K 的数据对比都有效但方向相反，按计划以 12K 下的结果为准"}
    vocab_votes = {}
    for name in ("vocab-fresh", "vocab-ep5", "vocab-ep20"):
        w = winner(by_name[name])
        if w is not None:
            vocab_votes[name] = "12k" if w.startswith("12k") else "8k"
    if not vocab_votes:
        vocab = {"choice": "12k", "basis": "三个词表对比都不有效，按计划默认 12K"}
    elif len(set(vocab_votes.values())) == 1:
        vocab = {"choice": next(iter(vocab_votes.values())), "basis": f"有效对比：{', '.join(sorted(vocab_votes))}"}
    elif "vocab-fresh" in vocab_votes:
        vocab = {"choice": vocab_votes["vocab-fresh"], "basis": "有效对比方向不一致，按计划以新数据下的结果为准"}
    else:
        vocab = {"choice": "12k", "basis": "5 轮与 20 轮下的词表结果相反、新数据下不有效，按计划默认 12K"}
    return {"data": data, "vocabulary": vocab}


def bpc(milestone, name):
    return (milestone.get(name) or {}).get("bpc")


def judge(runs, frozen, out, models=None, smoke=False, root=None):
    root = Path(root or Path(__file__).resolve().parents[3])
    runs, frozen, out = Path(runs), Path(frozen), Path(out)
    if out.exists():
        raise SystemExit(f"output exists, choose a new directory: {out}")
    if read_json(frozen / "manifest.json")["version"] != VERSION:
        raise SystemExit(f"{frozen} is not {VERSION}")
    problems, arms = [], {}
    for arm in ARMS:
        directory = model_dir(root, arm, models)
        full_scores, full_meta = check_arm(arm, runs, directory, "full", problems)
        abbrev_scores, abbrev_meta = (None, None)
        if (runs / arm / "abbrev.jsonl").exists():
            abbrev_scores, abbrev_meta = check_arm(arm, runs, directory, "abbrev", problems)
        arms[arm] = {"directory": directory, "full": full_scores, "full_meta": full_meta,
                     "abbrev": abbrev_scores, "abbrev_meta": abbrev_meta,
                     "training": training_record(arm, directory, problems, smoke)}
    tokenizers = {}
    for arm, record in arms.items():
        if record["full_meta"]:
            tokenizers.setdefault(ARMS[arm][0], set()).add(record["full_meta"].get("tokenizer_model_sha256"))
    if any(len(v) != 1 for v in tokenizers.values()) or len(set().union(*tokenizers.values())) != len(tokenizers):
        problems.append(f"tokenizer SHA256 per vocabulary is not one-to-one: {tokenizers}")
    abbrev_present = [a for a in ARMS if arms[a]["abbrev"]]
    if abbrev_present and len(abbrev_present) != len(ARMS):
        problems.append(f"abbreviated track scored for only {len(abbrev_present)} of {len(ARMS)} arms")
    if problems:
        raise SystemExit("judgement refused:\n  " + "\n  ".join(problems))

    out.mkdir(parents=True)
    (out / "scores").mkdir()
    tracks = ("full", "abbrev") if abbrev_present else ("full",)
    ranked = {}
    for track in tracks:
        ranked[("rime", track)] = run_scorer(frozen, out / "scores" / f"rime-{track}", track, name="rime-pool-order")
        for arm in ARMS:
            ranked[(arm, track)] = run_scorer(frozen, out / "scores" / f"{arm}-{track}", track, arms[arm][track], name=arm)
    ids = sorted(ranked[("rime", "full")][1])
    for (arm, track), (summary, rows) in ranked.items():
        if track == "full" and sorted(rows) != ids:
            raise SystemExit(f"{arm} covers different full-track items")
        if summary["fallbacks"]:
            raise SystemExit(f"{arm}/{track} has {summary['fallbacks']} items without scores")
    if len(ids) != ITEMS:
        raise SystemExit(f"expected {ITEMS} development items, found {len(ids)}")
    rime_top1 = ranked[("rime", "full")][0]["all"]["top1"]
    if rime_top1 != RIME_BASELINE_TOP1:
        raise SystemExit(f"Rime pool order gives {rime_top1}, the plan records {RIME_BASELINE_TOP1}")

    full = {arm: ranked[(arm, "full")] for arm in ARMS}
    top1 = {arm: full[arm][0]["all"]["top1"] for arm in ARMS}
    final = {arm: arms[arm]["training"]["milestones"][-1] for arm in ARMS}
    noise = {"top1": spread([top1[a] for a in SEEDS]),
             **{f"{n}_bpc": spread([bpc(final[a], n) for a in SEEDS]) for n in VALIDATIONS},
             "pairs": {f"{a} vs {b}": compare(full[a][1], full[b][1], ids)["top1"]
                       for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]}}
    seed_range = noise["top1"]["range"]

    contrasts = []
    for name, a, b, question in CONTRASTS:
        result = compare(full[a][1], full[b][1], ids)
        contrasts.append({"name": name, "a": a, "b": b, "question": question, "top1_a": top1[a], "top1_b": top1[b],
                          "difference": top1[b] - top1[a], "mcnemar_p": result["top1"]["exact_mcnemar_p"],
                          "top1": result["top1"], "top3": result["top3"]})
    for entry, adjusted in zip(contrasts, holm([c["mcnemar_p"] for c in contrasts])):
        entry["holm_p"] = adjusted
        entry["beyond_seed_range"] = abs(entry["difference"]) > seed_range
        entry["holm_significant"] = adjusted < ALPHA
        entry["effective"] = entry["beyond_seed_range"] and entry["holm_significant"]
    decision = decide(contrasts)

    descriptive = {
        "dose_response": [{"a": a, "b": b, "difference": top1[b] - top1[a],
                           **compare(full[a][1], full[b][1], ids)} for a, b in DESCRIPTIVE],
        "arms": {arm: {"vocabulary": ARMS[arm][0], "pool": ARMS[arm][1], "epochs": ARMS[arm][2],
                       "final_step": arms[arm]["training"]["step"],
                       "checkpoint_sha256": arms[arm]["full_meta"]["checkpoint_sha256"],
                       "full": {k: full[arm][0]["all"][k] for k in ("top1", "top1_rate", "top3", "pool_conditional_top1", "mrr")},
                       "full_by_source": {s: v["top1"] for s, v in full[arm][0]["by_source"].items()},
                       "full_without_overlap_top1": full[arm][0]["sensitivity_without_overlap"]["top1"],
                       "abbrev_top1": ranked[(arm, "abbrev")][0]["all"]["top1"] if "abbrev" in tracks else None,
                       "bpc": {n: bpc(final[arm], n) for n in VALIDATIONS},
                       "memorization_gap": (bpc(final[arm], "web") - bpc(final[arm], "probe")
                                            if bpc(final[arm], "web") is not None and bpc(final[arm], "probe") is not None else None),
                       "boundary_retokenized_items": arms[arm]["full_meta"].get("boundary_retokenized_items"),
                       "empty_pools": arms[arm]["full_meta"].get("empty_pools")}
                 for arm in ARMS},
        "learning_curves": {arm: [{"milestone": m["milestone"], "step": m["step"],
                                   "supervised_tokens": m.get("supervised_tokens"),
                                   **{n: bpc(m, n) for n in VALIDATIONS}}
                                  for m in arms[arm]["training"]["milestones"]] for arm in ARMS},
    }
    if "abbrev" in tracks:
        descriptive["abbrev_contrasts"] = [
            {"name": name, "a": a, "b": b,
             **compare(ranked[(a, "abbrev")][1], ranked[(b, "abbrev")][1], sorted(ranked[(a, "abbrev")][1]))["top1"]}
            for name, a, b, _ in CONTRASTS]
    result = {"format": "vimeml_zh13_judgement_v1", "plan": "docs/zh-CN/training-zh-v1-plan.md",
              "version": VERSION, "split": "development", "track": "full", "items": ITEMS,
              "smoke": smoke, "rime_baseline_top1": rime_top1, "alpha": ALPHA,
              "rule": "effective = |Top-1 difference| > range of the three 12k-fresh seeds and Holm-adjusted exact McNemar p < 0.05",
              "noise_band": noise, "contrasts": contrasts, "decision": decision, "descriptive": descriptive,
              "inputs": {arm: {"full_scores": str(arms[arm]["full"]), "abbrev_scores": str(arms[arm]["abbrev"]) if arms[arm]["abbrev"] else None,
                               "model_dir": str(arms[arm]["directory"])} for arm in ARMS}}
    (out / "judgement.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "report.md").write_text(report(result), encoding="utf-8", newline="\n")
    return result


def _p(value):
    return f"{value:.3g}" if value is not None else "—"


def _f(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "—"


def report(result):
    noise, arms = result["noise_band"], result["descriptive"]["arms"]
    lines = [f"# vimeml-tiny-zh-v1 判定{'（冒烟，不是正式结果）' if result['smoke'] else ''}", "",
             f"按 `{result['plan']}` 预注册的规则，由 `scripts/benchmarks/judge_zh13.py` 生成。"
             f"{result['version']} development 全拼轨道 {result['items']} 条，精确 Top-1；Rime 第一候选 {result['rime_baseline_top1']}。", "",
             "## 结论", "",
             f"- 数据：**{result['decision']['data']['choice'] or '未定'}**。{result['decision']['data']['basis']}。",
             f"- 词表：**{result['decision']['vocabulary']['choice'] or '未定'}**。{result['decision']['vocabulary']['basis']}。", "",
             "## 噪声带（三个 12k-fresh seed）", "",
             f"Top-1：{', '.join(map(str, noise['top1']['values']))}；均值 {noise['top1']['mean']:.1f}，"
             f"极差 **{noise['top1']['range']}**，标准差 {noise['top1']['sd']:.2f}。", ""]
    for name in VALIDATIONS:
        s = noise[f"{name}_bpc"]
        if s:
            lines.append(f"- {name} BPC：均值 {s['mean']:.4f}，极差 {s['range']:.4f}，标准差 {s['sd']:.4f}")
    lines += ["", "| seed 对 | +/− | McNemar p | sign-flip p |", "| --- | ---: | ---: | ---: |"]
    for pair, t in noise["pairs"].items():
        lines.append(f"| {pair} | +{t['gain']}/−{t['loss']} | {_p(t['exact_mcnemar_p'])} | {_p(t['cluster_sign_flip_p'])} |")
    lines += ["", "## 五个判定对比（Holm，族 α = 0.05）", "",
              "| 对比 | A | B | Top-1 A | Top-1 B | B − A | +/− | McNemar p | Holm p | sign-flip p | 超过极差 | 有效 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |"]
    for c in result["contrasts"]:
        t = c["top1"]
        lines.append(f"| {c['question']} | `{c['a']}` | `{c['b']}` | {c['top1_a']} | {c['top1_b']} | {c['difference']:+d} | "
                     f"+{t['gain']}/−{t['loss']} | {_p(c['mcnemar_p'])} | {_p(c['holm_p'])} | {_p(t['cluster_sign_flip_p'])} | "
                     f"{'是' if c['beyond_seed_range'] else '否'} | **{'是' if c['effective'] else '否'}** |")
    lines += ["", "以下全部只描述，不参与判定。", "", "## 各臂", "",
              "| 臂 | 词表 | 池 | 轮 | Top-1 | Top-3 | 池条件 Top-1 | 简拼 Top-1 | web BPC | chat BPC | probe BPC | 记忆差 |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, a in arms.items():
        lines.append(f"| `{arm}` | {a['vocabulary'] // 1024}K | {a['pool']} | {a['epochs']} | {a['full']['top1']} | {a['full']['top3']} | "
                     f"{_f(a['full']['pool_conditional_top1'], 3)} | {a['abbrev_top1'] if a['abbrev_top1'] is not None else '—'} | "
                     f"{_f(a['bpc']['web'])} | {_f(a['bpc']['chat'])} | {_f(a['bpc']['probe'])} | {_f(a['memorization_gap'])} |")
    lines += ["", "记忆差 = validation 网页 BPC − P20 记忆抽样 BPC，越大说明训练文档比没见过的文档容易得越多。", "",
              "## 剂量-反应（ep5）", "", "| A | B | B − A | +/− | McNemar p |", "| --- | --- | ---: | ---: | ---: |"]
    for d in result["descriptive"]["dose_response"]:
        t = d["top1"]
        lines.append(f"| `{d['a']}` | `{d['b']}` | {d['difference']:+d} | +{t['gain']}/−{t['loss']} | {_p(t['exact_mcnemar_p'])} |")
    if "abbrev_contrasts" in result["descriptive"]:
        lines += ["", "## 简拼轨道的同五个对比", "", "| A | B | 净差 | +/− | McNemar p |", "| --- | --- | ---: | ---: | ---: |"]
        for c in result["descriptive"]["abbrev_contrasts"]:
            lines.append(f"| `{c['a']}` | `{c['b']}` | {c['net']:+d} | +{c['gain']}/−{c['loss']} | {_p(c['exact_mcnemar_p'])} |")
    lines += ["", "## 各来源 Top-1", "", "| 臂 | crosswoz | risawoz | kdconv | 去重叠后 Top-1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, a in arms.items():
        s = a["full_by_source"]
        lines.append(f"| `{arm}` | {s['crosswoz']} | {s['risawoz']} | {s['kdconv']} | {a['full_without_overlap_top1']} |")
    lines += ["", "学习曲线（每 1/20 的 web/chat/probe BPC）见 `judgement.json` 的 `descriptive.learning_curves`。", ""]
    return "\n".join(lines)
