"""Conservative sentence segmentation for Japanese paragraphs."""

import re


PAIRS = {
    "「": "」",
    "『": "』",
    "（": "）",
    "(": ")",
    "[": "]",
    "【": "】",
    "“": "”",
}
ENDINGS = set("。！？!?")
URL = re.compile(r"https?://[^\s「」『』（）()\[\]【】“”。！？]+")


def split_sentences(text: str) -> list[dict]:
    if "\n" in text or "\r" in text:
        raise ValueError("请先处理换行，输入应为单个段落。")

    results = []
    stack = []
    issues = set()
    start = 0
    url_ranges = [match.span() for match in URL.finditer(text)]

    def emit(end, terminated):
        nonlocal start

        raw = text[start:end]
        left = start + len(raw) - len(raw.lstrip())
        right = end - (len(raw) - len(raw.rstrip()))

        if left < right:
            results.append({
                "text": text[left:right],
                "start": left,
                "end": right,
                "terminated": terminated,
                "flags": sorted(issues),
            })

        start = end
        issues.clear()

    for index, char in enumerate(text):
        # Question marks in URLs are not sentence boundaries.
        if any(left <= index < right for left, right in url_ranges):
            continue

        if char in PAIRS:
            stack.append(PAIRS[char])
        elif char in PAIRS.values():
            if stack and stack[-1] == char:
                stack.pop()
            else:
                issues.add("unmatched_closing_bracket")

        if char in ENDINGS and not stack:
            # Keep consecutive punctuation together, e.g. !?
            if index + 1 < len(text) and text[index + 1] in ENDINGS:
                continue
            emit(index + 1, True)

    if stack:
        issues.add("unclosed_bracket")

    tail = text[start:].strip().rstrip(
        "".join(PAIRS.values())
    ).rstrip()
    emit(len(text), bool(tail) and tail[-1] in ENDINGS)

    return results

CONTINUATIONS = (
    "のが", "のを", "ので", "のに", "なくて", "して", "ながら",
    "という", "による", "であり", "ても",
    "は", "が", "を", "に", "と", "や", "の", "し", "も",
)
LIST_ITEM = re.compile(r"^(?:[-*・●■◆◇]\s*|[0-9０-９]+[.)．、])")


def join_reason(left: str, right: str):
    if LIST_ITEM.match(left) or LIST_ITEM.match(right):
        return None

    tail = split_sentences(left)[-1]
    if "unclosed_bracket" in tail["flags"]:
        combined = split_sentences(left + right)
        if not any(sentence["flags"] for sentence in combined):
            return "unclosed_bracket"
        return None
    if tail["terminated"]:
        return None
    if left.endswith(("、", "，", ",")):
        return "trailing_comma"
    if left.endswith(CONTINUATIONS):
        return "continuation_suffix"
    if right.startswith(("を", "について", "によって", "として")):
        return "continuation_prefix"
    return None



def closes_brackets_ahead(left: str, blocks: list[dict], start: int, max_chars: int, boundary_decisions=None) -> bool:
    """Look ahead for a balanced close without crossing a structural boundary."""
    combined = left
    previous_index = blocks[start]["line_index"] - 1
    for position in range(start, len(blocks)):
        block = blocks[position]
        if boundary_decisions and position > 0 and boundary_decisions.get((blocks[position - 1]["id"], block["id"])) == "separate":
            return False
        text = block["text"].strip()
        index = block["line_index"]
        if (
            index != previous_index + 1 or block["action"] != "keep"
            or not text or LIST_ITEM.match(text)
        ):
            return False
        previous_index = index
        separator = " " if (
            combined[-1].isascii() and text[0].isascii()
            and combined[-1].isalnum() and text[0].isalnum()
        ) else ""
        if len(combined) + len(separator) + len(text) > max_chars:
            return False
        combined += separator + text
        sentences = split_sentences(combined)
        if not any(sentence["flags"] for sentence in sentences):
            return True
        if any("unmatched_closing_bracket" in sentence["flags"] for sentence in sentences):
            return False
    return False


def restore_linebreaks(
    blocks: list[dict], max_chars: int = 2048, boundary_decisions=None
) -> list[dict]:
    boundary_decisions = boundary_decisions or {}
    if any(value not in {"join", "separate"} for value in boundary_decisions.values()):
        raise ValueError("Unknown approved boundary decision.")
    paragraphs = []
    current = None
    previous_index = None

    def flush():
        nonlocal current
        if current is not None:
            paragraphs.append(current)
            current = None

    for position, block in enumerate(blocks):
        index = block["line_index"]
        if not isinstance(index, int) or index < 0:
            raise ValueError("line_index 必须是非负整数。")
        if previous_index is not None and index <= previous_index:
            raise ValueError("文本块必须按原始行号严格递增。")

        consecutive = (
            previous_index is not None and index == previous_index + 1
        )
        previous_index = index

        action = block["action"]
        if action not in {"keep", "drop", "review"}:
            raise ValueError("未知的 action。")

        text = block["text"].strip()
        if action != "keep" or not text:
            flush()
            continue
        if not consecutive:
            flush()

        reason = join_reason(current["text"], text) if current else None
        decision = boundary_decisions.get((current["parts"][-1]["block_id"], block["id"])) if current else None
        if decision == "separate":
            reason = None
        elif decision == "join":
            if LIST_ITEM.match(current["text"]) or LIST_ITEM.match(text):
                raise ValueError("Approved join cannot cross a list/title boundary.")
            reason = "approved_annotation"
        if (
            current and reason is None and decision != "separate"
            and not LIST_ITEM.match(current["text"])
            and not LIST_ITEM.match(text)
            and "unclosed_bracket" in split_sentences(current["text"])[-1]["flags"]
            and closes_brackets_ahead(current["text"], blocks, position, max_chars, boundary_decisions)
        ):
            reason = "unclosed_bracket"
        separator = ""

        # Preserve the boundary between Latin words or numbers.
        if reason:
            last = current["text"][-1]
            first = text[0]
            if (
                last.isascii() and first.isascii()
                and last.isalnum() and first.isalnum()
            ):
                separator = " "

        if reason and (
            len(current["text"]) + len(separator) + len(text) > max_chars
        ):
            if decision == "join":
                raise ValueError("Approved join exceeds max_chars; review the complete chain.")
            current["flags"].append("join_length_limit")
            reason = None

        if reason:
            offset = len(current["text"]) + len(separator)
            current["joins"].append({
                "left_block": current["parts"][-1]["block_id"],
                "right_block": block["id"],
                "reason": reason,
            })
            current["text"] += separator + text
            current["parts"].append({
                "block_id": block["id"],
                "start": offset,
                "end": offset + len(text),
            })
        else:
            flush()
            current = {
                "text": text,
                "parts": [{
                    "block_id": block["id"],
                    "start": 0,
                    "end": len(text),
                }],
                "joins": [],
                "flags": [],
            }

    flush()
    return paragraphs
