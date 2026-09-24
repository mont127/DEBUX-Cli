"""The Debux directive contract: parsing it, and shaping the context the model expects.

Debux replies are reasoning followed by exactly one directive. The CLI acts on that directive, so
it is an interface rather than a writing style. A reply with no directive is a failure and is
handled as one rather than shown as prose.
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRECTIVE = re.compile(r"^[ \t]*(NEED|SEARCH|DIAGNOSIS|INSUFFICIENT)[ \t]*:(.*)$", re.M | re.S)

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
        line = re.sub(r"^[$#]\s*", "", line)
        line = line.strip("`").strip()
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
