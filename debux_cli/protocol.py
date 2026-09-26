"""The Debux directive contract: parsing it, and shaping the context the model expects.

Debux replies are reasoning followed by exactly one directive. The CLI acts on that directive, so
it is an interface rather than a writing style. A reply with no directive is a failure and is
handled as one rather than shown as prose.
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRECTIVE = re.compile(
    r"^[ \t]*(NEED|SEARCH|FETCH|DIAGNOSIS|PROCEDURE|INSUFFICIENT)[ \t]*:(.*)$", re.M | re.S)

URL = re.compile(r"https?://[^\s`'\"<>)\]]+")

# The model rarely asks to search on its own (measured 0/10 on the benchmark), but it does say
# plainly when it is out of its depth. These are the phrasings that mean "this depends on a
# version or vendor fact I do not have", which is exactly when a lookup helps.
UNSURE = re.compile(
    r"\b(?:I (?:do not|don't) know|not (?:certain|sure) (?:which|what|whether)"
    r"|depends on the (?:exact )?(?:version|release|distribution|vendor|provider)"
    r"|varies by (?:version|distribution|vendor|provider)"
    r"|consult (?:the )?(?:official )?(?:documentation|docs|release notes|vendor)"
    r"|check (?:the )?(?:official )?(?:documentation|docs|release notes|changelog)"
    r"|would need to (?:look ?up|verify|confirm) (?:the )?(?:exact|current|specific)"
    r"|I (?:will|won't|will not) invent)\b", re.I)


def urls(text):
    return URL.findall(text or "")


def sounds_unsure(text):
    return bool(UNSURE.search(text or ""))

# Must match the corpus builder's --window. The model was trained on the system prompt, the
# original symptom, and the last message only. A full transcript is a context shape it has never
# seen, and it degrades accordingly. Keeping the symptom matters: without it a late turn loses
# track of what is being debugged.
WINDOW = 1


def system_prompt(path=None):
    p = path or os.path.join(HERE, "system_prompt.txt")
    with open(p) as fh:
        return fh.read().strip()


def windowed(messages, window=WINDOW):
    """system + original symptom + the last `window` messages."""
    if window <= 0 or len(messages) <= 2 + window:
        return messages
    return messages[:2] + messages[2:][-window:]


def split(text):
    """(kind, directive_block, reasoning) for a reply, or (None, None, text).

    The FIRST directive wins, not the last. A model that reasons its way to NEED and then appends
    a speculative DIAGNOSIS has still overclaimed; taking the last one would hide exactly that.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = DIRECTIVE.search(text)
    if not m:
        return None, None, text
    return m.group(1).upper(), text[m.start():], text[:m.start()]


def commands(block, limit=6):
    """Pull runnable commands out of a NEED block, dropping prose and decoration."""
    if ":" not in block:
        return []
    out = []
    for raw in block.split(":", 1)[1].splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        # ```bash opens a fence; the language tag is not a command
        if line.startswith("```"):
            continue
        if line.startswith("#"):
            continue
        line = re.sub(r"^\$\s*", "", line)
        # The model often writes: 1. `cmd --flag` (why this command matters).
        # Prefer whatever is inside backticks; otherwise cut at the first " (" or " to ".
        m = re.search(r"`([^`]+)`", line)
        if m:
            line = m.group(1).strip()
        else:
            line = line.strip("`").strip()
            line = re.split(r"\s+\((?:or|and|to|adjust|replace|check|test|see)\b|\s+(?:to|which|so that)\s+(?:identify|see|confirm|check|test)\b",
                            line)[0].strip()
        line = line.rstrip("`").strip().rstrip(".")
        if not line or line.endswith(":"):
            continue
        # a sentence rather than a command
        if " " in line and not re.match(r"^[a-zA-Z_./~][\w./-]*\s", line):
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return out


def query(block):
    """The search string from a SEARCH block."""
    if ":" not in block:
        return ""
    return block.split(":", 1)[1].strip().splitlines()[0].strip() if block.split(":", 1)[1].strip() else ""
