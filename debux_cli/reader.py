"""Reading pasted terminal output without making the user mark the end of it.

Command output is multi-line and full of blank lines, so a blank line cannot end a block. The
original design required a lone "." which works but is a thing to remember, and forgetting it
looks exactly like the program hanging.

A paste arrives as a burst: many lines within milliseconds of each other, because the terminal
writes the whole clipboard at once. Typing does not do that. So a burst followed by a short
quiet period is a finished paste and can be submitted on its own. Typed input keeps the explicit
terminator, since a human pausing to think must not be cut off mid-thought.

"." and Ctrl-D always submit, whatever the timing.
"""
import select
import sys

BURST_GAP = 0.12     # lines closer together than this came from a paste, not a keyboard
QUIET_AFTER = 0.55   # silence this long after a burst means the paste is done


def read_block(prompt, out=print):
    """Collect pasted output. Returns the text, or "" if nothing was given."""
    out(prompt)
    if not sys.stdin.isatty():
        return _read_piped()

    lines, pasted, last = [], False, None
    while True:
        timeout = QUIET_AFTER if (pasted and lines) else None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except (OSError, ValueError):
            return _read_piped(lines)

        if not ready:
            # quiet after a burst: the paste has finished
            return "\n".join(lines).strip()

        line = sys.stdin.readline()
        if line == "":              # Ctrl-D
            return "\n".join(lines).strip()
        line = line.rstrip("\n")
        if line.strip() == ".":
            return "\n".join(lines).strip()

        now = _now()
        if last is not None and (now - last) < BURST_GAP:
            pasted = True
        last = now
        lines.append(line)
    # unreachable


def _read_piped(lines=None):
    lines = list(lines or [])
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _now():
    import time
    return time.monotonic()


def read_line_or_paste(prompt):
    """Read one typed line, or the whole thing if the user pasted several.

    People paste the problem and the command output together at the first prompt. A plain
    input() takes the first line only and the rest leaks into the next read, so the model is
    asked to diagnose a symptom with its evidence torn off. Read the first line, then take
    anything that is already waiting behind it.
    """
    try:
        first = input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise
    if not sys.stdin.isatty():
        return first
    lines = [first]
    while True:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], BURST_GAP)
        except (OSError, ValueError):
            break
        if not ready:
            break
        line = sys.stdin.readline()
        if line == "":
            break
        line = line.rstrip("\n")
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()
