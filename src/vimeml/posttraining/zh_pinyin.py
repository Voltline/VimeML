"""Pinyin-conditioned generation samples for the zh three-task model.

Sequence (zh-v1 control ids 4/5/6, the layout of the Japanese conversion data):
    [2, 4] + context + [5] + keys + [6] + output + [3]
Loss covers output and EOS. keys are raw keystrokes written the way they are typed,
continuously: full pinyin (zhongguorenmin), jieba words cut to their first letters
(zxiazhouyishangwudeiq), or every syllable cut to its first letter. An apostrophe
before a vowel-initial syllable (xi'an) is typed with apostrophe_probability. ü is v.

Kinds:
  full      every syllable spelled in full
  mixed     each jieba word in full or as initials, at least one of each
  initials  every syllable as its first letter (zh/ch/sh kept with keep_zh_probability)
  error     full or mixed keys with misread syllables, output still the correct text.
            Sources: a listed proper-noun / common-word misreading, a polyphone read
            with its single-character default or another heteronym, a fuzzy-pinyin
            confusion (z/zh, n/l, an/ang, ...). Misread syllables are always spelled
            in full, and every misreading must be a standard syllable.

Readings come from pypinyin over the whole Han run (phrase-aware), then are sliced to the
target span; a run with any character outside the standard syllable set is skipped.
"""
from functools import lru_cache
import logging
import re
import warnings

HAN_RUN = re.compile(r"[一-鿿]{2,}")
VOWELS = ("a", "e", "o")
KINDS = ("full", "mixed", "initials", "error")
CONTROL = (4, 5, 6)
# Function characters whose other readings nobody types by mistake (的 -> di, 了 -> liao).
FUNCTION_CHARS = frozenset("的了着得地和都还么呢吧啊哪那个们子头一不")

# word -> a misreading people type (toneless, space-separated syllables). An entry is
# used only when pypinyin's phrase reading differs from it and has the same length.
MISREADINGS = {
    "蚌埠": "bang bu", "六安": "liu an", "番禺": "fan yu", "亳州": "hao zhou", "莘县": "xin xian",
    "枞阳": "cong yang", "歙县": "xi xian", "盱眙": "yu tai", "浚县": "jun xian", "洪洞": "hong dong",
    "犍为": "jian wei", "蔚县": "wei xian", "铅山": "qian shan", "乐亭": "le ting", "尉迟": "wei chi",
    "单于": "dan yu", "万俟": "wan si", "长孙": "chang sun", "秘鲁": "mi lu", "龟兹": "gui zi",
    "吐蕃": "tu fan", "阿房宫": "a fang gong", "大月氏": "da yue shi", "荨麻疹": "qian ma zhen",
    "粳米": "geng mi", "酵母": "xiao mu", "造诣": "zao zhi", "憎恶": "ceng wu", "吊唁": "diao yin",
    "亲家": "qin jia", "关卡": "guan ka", "角色": "jiao se", "机械": "ji jie", "混淆": "hun yao",
    "酗酒": "xiong jiu", "纤维": "qian wei", "皈依": "ban yi", "包扎": "bao za", "炽热": "zhi re",
    "弄堂": "nong tang", "栅栏": "shan lan", "隽永": "jun yong", "莞尔": "guan er", "恪守": "ge shou",
    "赝品": "ying pin", "肄业": "si ye", "拮据": "ji ju", "粗犷": "cu kuang", "刽子手": "kuai zi shou",
    "狙击": "zu ji", "羸弱": "ying ruo", "汲取": "xi qu", "参差": "can cha", "说服": "shui fu",
    "解数": "jie shu", "呱呱坠地": "gua gua zhui di", "一叶扁舟": "yi ye bian zhou", "大夫": "da fu",
    "内讧": "nei hong", "惬意": "xia yi", "暴殄天物": "bao zhen tian wu", "谄媚": "xian mei",
    "纨绔": "wan kua", "瞠目": "tang mu", "不啻": "bu di", "编纂": "bian zuan", "教诲": "jiao mei",
    "症结": "zheng jie", "复辟": "fu pi", "唾手可得": "chui shou ke de", "鞭笞": "bian tai",
}

INITIAL_SWAPS = (("zh", "z"), ("ch", "c"), ("sh", "s"), ("z", "zh"), ("c", "ch"), ("s", "sh"),
                 ("n", "l"), ("l", "n"), ("f", "h"), ("h", "f"),
                 ("r", "l"), ("l", "r"))
FINAL_SWAPS = (("iang", "ian"), ("ian", "iang"), ("uang", "uan"), ("uan", "uang"), ("ang", "an"),
               ("an", "ang"), ("eng", "en"), ("en", "eng"), ("ing", "in"), ("in", "ing"))


def quiet_imports():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import jieba
    jieba.setLogLevel(logging.WARNING)
    jieba.initialize()
    return jieba


def standard_syllables():
    """Every toneless reading pypinyin gives a basic-block character, minus vowel-less ones."""
    from pypinyin import Style, pinyin
    found = set()
    for code in range(0x4E00, 0xA000):
        for value in pinyin(chr(code), heteronym=True, style=Style.NORMAL, errors="ignore")[:1]:
            found.update(value)
    return frozenset(s for s in found if s.isascii() and s.isalpha() and any(v in s for v in "aeiouv"))


def common_readings(share, valid):
    """char -> readings holding >= share of its occurrences in pypinyin's phrase table."""
    from collections import Counter, defaultdict
    from pypinyin import Style
    from pypinyin.phrases_dict import phrases_dict
    from pypinyin.style import convert
    counts = defaultdict(Counter)
    for word, readings in phrases_dict.items():
        if len(word) == len(readings):
            for char, reading in zip(word, readings):
                counts[char][convert(reading[0], Style.NORMAL, strict=False)] += 1
    out = {}
    for char, c in counts.items():
        total = sum(c.values())
        kept = tuple(sorted(r for r, n in c.items() if n >= share * total and r in valid))
        if len(kept) > 1:
            out[char] = kept
    return out


class Pinyin:
    def __init__(self, config, syllables=None):
        from pypinyin import Style, lazy_pinyin, pinyin
        self.jieba = quiet_imports()
        self.style, self.lazy, self.pinyin = Style.NORMAL, lazy_pinyin, pinyin
        self.valid = syllables or standard_syllables()
        self.config = config
        self.lengths = [(int(a), int(b), float(w)) for a, b, w in config["target_lengths"]]
        self.misreadings = {}
        for word, wrong in MISREADINGS.items():
            wrong = wrong.split()
            right = self.lazy(word, style=self.style)
            if len(wrong) == len(word) == len(right) and wrong != right and all(s in self.valid for s in wrong):
                self.misreadings[word] = wrong
        self.misread_pattern = re.compile("|".join(sorted(map(re.escape, self.misreadings), key=len, reverse=True)))
        self.default_reading = lru_cache(maxsize=1 << 16)(self._default_reading)
        self.common = common_readings(config["common_reading_share"], self.valid)

    def _default_reading(self, char):
        return self.lazy(char, style=self.style)[0]

    # ---- span ----------------------------------------------------------------------

    def target_length(self, rng):
        total = sum(w for _, _, w in self.lengths)
        x = rng.random() * total
        for low, high, weight in self.lengths:
            x -= weight
            if x < 0:
                return rng.randint(low, high)
        return self.lengths[-1][1]

    def span(self, rng, text, require=None):
        """(run start, run text, word boundaries, span start, span end) in run coordinates."""
        runs = [m for m in HAN_RUN.finditer(text)]
        if require is not None:
            runs = [m for m in runs if m.start() <= require[0] and require[1] <= m.end()]
        if not runs:
            return None
        # Draw the length first and prefer runs long enough for it, so clause-sized runs
        # do not pull every target short; fall back to the longest run.
        wanted = self.target_length(rng)
        fits = [m for m in runs if m.end() - m.start() >= wanted]
        if fits:
            run = rng.choices(fits, weights=[m.end() - m.start() for m in fits])[0]
        else:
            run = max(runs, key=lambda m: m.end() - m.start())
        run_text = run.group()
        bounds, cursor = [0], 0
        for word in self.jieba.lcut(run_text):
            cursor += len(word)
            bounds.append(cursor)
        size = min(wanted, len(run_text))
        if require is not None:
            low, high = require[0] - run.start(), require[1] - run.start()
            starts = [b for b in bounds[:-1] if b <= low]
            start = rng.choice(starts[-3:])
            end = next(b for b in bounds if b >= min(max(high, start + size), len(run_text)))
        else:
            starts = [b for b in bounds[:-1] if len(run_text) - b >= size] or [0]
            start = rng.choice(starts)
            end = next(b for b in bounds if b >= start + size)
        if end - start < 2:
            return None
        return run.start(), run_text, bounds, start, end

    # ---- keys ----------------------------------------------------------------------

    def initial(self, rng, syllable):
        if syllable[:2] in ("zh", "ch", "sh") and rng.random() < self.config["keep_zh_probability"]:
            return syllable[:2]
        return syllable[0]

    def join(self, rng, units):
        """units: (text, spelled_in_full). Apostrophe only before a vowel-initial full syllable."""
        out = []
        for i, (text, full) in enumerate(units):
            if i and full and text[0] in VOWELS and rng.random() < self.config["apostrophe_probability"]:
                out.append("'")
            out.append(text)
        return "".join(out)

    def abbreviation_mask(self, rng, words):
        """words: list of (start, end) syllable ranges. True = cut to initials."""
        if len(words) == 1:
            start, end = words[0]
            words = [(i, i + 1) for i in range(start, end)]
        p = rng.uniform(*self.config["mixed_abbreviation_range"])
        cut = [rng.random() < p for _ in words]
        if all(cut) or not any(cut):
            flip = rng.randrange(len(words))
            cut[flip] = not cut[flip]
        mask = []
        for (start, end), c in zip(words, cut):
            mask += [c] * (end - start)
        return mask

    # ---- errors --------------------------------------------------------------------

    def fuzzy(self, syllable):
        options = []
        for a, b in INITIAL_SWAPS:
            if syllable.startswith(a) and not (len(a) == 1 and a in "zcs" and syllable[1:2] == "h"):
                options.append(b + syllable[len(a):])
        for a, b in FINAL_SWAPS:
            if syllable.endswith(a):
                options.append(syllable[:-len(a)] + b)
                break
        return [s for s in options if s in self.valid and s != syllable]

    def misread(self, rng, chars, syllables, span_text):
        """Return (new syllables, changed positions, sources) or None."""
        syllables, changed, sources = list(syllables), set(), []
        for m in self.misread_pattern.finditer(span_text):
            if rng.random() < self.config["error_listed_probability"]:
                wrong = self.misreadings[m.group()]
                if syllables[m.start():m.end()] != wrong:
                    syllables[m.start():m.end()] = wrong
                    changed.update(range(m.start(), m.end()))
                    sources.append("listed")
        extra = 0 if sources else (1 if rng.random() < self.config["error_single_probability"] else 2)
        for _ in range(extra):
            poly, fuzzy = [], []
            for i, (char, s) in enumerate(zip(chars, syllables)):
                if i in changed or char in FUNCTION_CHARS:
                    continue
                # The single-character default is the usual misreading (银行 -> yinxing);
                # other heteronyms come second.
                default = self.default_reading(char)
                others = sorted(set(self.common.get(char, ())) - {s, default})
                if default != s and default in self.valid and (not others or rng.random() < 0.7):
                    poly.append((i, [default]))
                elif others:
                    poly.append((i, others))
                options = self.fuzzy(s)
                if options:
                    fuzzy.append((i, options))
            order = [("polyphone", poly), ("fuzzy", fuzzy)]
            if rng.random() >= self.config["error_polyphone_share"]:
                order.reverse()
            for name, pool in order:
                if pool:
                    i, options = rng.choice(pool)
                    syllables[i] = rng.choice(options)
                    changed.add(i)
                    sources.append(name)
                    break
        if not changed:
            return None
        return syllables, changed, sources

    # ---- sample --------------------------------------------------------------------

    def sample(self, rng, text, kind, require=None):
        """(context, keys, output, info) or None. text is one paragraph or utterance."""
        found = self.span(rng, text, require)
        if found is None:
            return None
        run_start, run_text, bounds, start, end = found
        readings = self.lazy(run_text, style=self.style, errors=lambda x: [None] * len(x))
        if len(readings) != len(run_text) or any(s not in self.valid for s in readings):
            return None
        output = run_text[start:end]
        syllables = readings[start:end]
        words = [(max(a, start) - start, min(b, end) - start) for a, b in zip(bounds, bounds[1:]) if a < end and b > start]
        info = {"kind": kind}
        if kind == "error":
            base = "full" if rng.random() < self.config["error_full_share"] else "mixed"
            result = self.misread(rng, output, syllables, output)
            if result is None:
                return None
            syllables, changed, info["errors"] = result
            mask = self.abbreviation_mask(rng, words) if base == "mixed" else [False] * len(syllables)
            mask = [m and i not in changed for i, m in enumerate(mask)]
            info["base"] = base
        elif kind == "full":
            mask = [False] * len(syllables)
        elif kind == "initials":
            mask = [True] * len(syllables)
        elif kind == "mixed":
            mask = self.abbreviation_mask(rng, words)
        else:
            raise ValueError(kind)
        units = [(self.initial(rng, s), False) if m else (s, True) for s, m in zip(syllables, mask)]
        keys = units[0][0] + "".join(u for u, _ in units[1:]) if kind == "initials" else self.join(rng, units)
        context = ""
        if rng.random() >= self.config["empty_context_probability"]:
            before = text[:run_start + start]
            context = before[len(before) - rng.randint(1, self.config["max_context_characters"]):] if before else ""
        return context, keys, output, info


def encode(sp, context, keys, output, limit=128):
    """ids for [2,4]+ctx+[5]+keys+[6]+out+[3] and the first supervised index, or None."""
    r, out = sp.encode(keys), sp.encode(output)
    if sp.decode(out) != output or sp.decode(r) != keys or any(i in CONTROL for i in r + out):
        return None
    budget = limit + 1 - (len(r) + len(out) + 5)
    if budget < 0:
        return None
    ctx, removed = sp.encode(context), 0
    while len(ctx) > budget or any(i in CONTROL for i in ctx):
        removed += 1
        ctx = sp.encode(context[removed:])
    ids = [2, 4] + ctx + [5] + r + [6] + out + [3]
    return ids, 4 + len(ctx) + len(r), removed
