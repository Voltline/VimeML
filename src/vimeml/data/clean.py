"""Conservative text-block cleaning rules."""

import re
import unicodedata


URL_ONLY = re.compile(r"(?:[→↗➜>]\s*)?https?://\S+")
PAIRED_DECORATION = re.compile(r"([◇◆■□★☆]{2,})(.+?)\1")
NAVIGATION_SUFFIX = re.compile(r"[。！？!?]\s*Next\s*$")
DECORATION_CHARS = set(" -=_*#~・◇◆■□★☆→←↑↓/\\|─━")


def clean_block(original: str) -> dict:
    changes = []
    flags = []

    text = unicodedata.normalize("NFC", original)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff")

    # Replace controls with spaces to avoid joining unrelated words.
    text = "".join(
        " " if unicodedata.category(char) == "Cc"
        and char not in "\n\t" else char
        for char in text
    ).strip()

    if text != original:
        changes.append("basic_normalization")

    action = "keep"
    reason = "no_definite_noise"

    if not text:
        action, reason = "drop", "empty"
    elif URL_ONLY.fullmatch(text):
        action, reason = "drop", "standalone_url"
    elif all(char in DECORATION_CHARS for char in text):
        action, reason = "drop", "decoration_only"
    else:
        match = PAIRED_DECORATION.fullmatch(text)
        if match:
            text = match.group(2).strip()
            changes.append("paired_decoration_removed")

        if "\ufffd" in text:
            flags.append("replacement_character")
        if NAVIGATION_SUFFIX.search(text):
            flags.append("possible_navigation_suffix")

        if flags:
            action, reason = "review", "suspected_noise"

    return {
        "text": text,
        "action": action,
        "reason": reason,
        "changes": changes,
        "flags": flags,
    }