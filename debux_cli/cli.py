"""Debux CLI - an evidence-driven Linux and DevOps debugging assistant.

You have a broken server. Debux asks what to run, reads what you paste back, and reasons about
what that output rules in and out until it can name a cause or say honestly that it cannot.

Debux never executes anything on your machine. You run the commands. A debugging tool with no
execution path has no blast radius, and you see every command before it runs.
"""
import argparse
import json
import os
import shlex
import sys
import textwrap
import time

from . import backends, protocol, reader, tools

HOME = os.path.expanduser("~/.debux")
CONF = os.path.join(HOME, "config.json")


class C:
    d = "\033[2m"; b = "\033[1m"; r = "\033[0m"
    cy = "\033[36m"; ye = "\033[33m"; gr = "\033[32m"; rd = "\033[31m"; ma = "\033[35m"

    @classmethod
    def off(cls):
        for k, v in list(vars(cls).items()):
            if isinstance(v, str) and v.startswith("\033"):
                setattr(cls, k, "")


def wrap(text, indent="  ", width=96):
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        sub = indent + ("  " if para.lstrip().startswith(("-", "*")) else "")
        out.extend(textwrap.wrap(para, width=width, initial_indent=indent,
                                 subsequent_indent=sub) or [""])
    return "\n".join(out)


def load_conf():
    try:
        with open(CONF) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_conf(conf):
    os.makedirs(HOME, exist_ok=True)
    try:
        with open(CONF, "w") as fh:
            json.dump(conf, fh, indent=1)
        os.chmod(CONF, 0o600)   # it can hold a /connect token
    except OSError as e:
        print(f"{C.ye}could not save config: {e}{C.r}")


HELP = [
    ("/connect <url> [--token t]", "route the model to a remote OpenAI-compatible server"),
    ("/connect status", "check the connected server"),
    ("/connect disconnect", "go back to the local backend"),
    ("/search <query>", "search the web yourself and add the results"),
    ("/fetch <url>", "fetch a page and add it as evidence"),
    ("/upload <path>", "add a file - a log, a unit file, a config"),
    ("/paste", "paste a block of output without being asked"),
    ("/model <name>", "switch model"),
    ("/reset", "start a new problem, clearing the conversation"),
    ("/save [path]", "write the transcript to a file"),
    ("/help", "this list"),
    ("/quit", "exit"),
]


class Session:
    def __init__(self, backend, system, transcript=None, no_search=False, auto_tools=True):
        self.backend = backend
        self.system = system
        self.messages = [{"role": "system", "content": system}]
        self.transcript = transcript
        self.no_search = no_search
        self.auto_tools = auto_tools
        self.local = backend          # restored by /connect disconnect
        self.log = []

    # ---------------------------------------------------------------- plumbing

    def record(self, role, content, kind=None):
        self.log.append({"role": role, "kind": kind, "content": content, "t": time.time()})
        if self.transcript:
            try:
                with open(self.transcript, "a") as fh:
                    fh.write(json.dumps(self.log[-1]) + "\n")
            except OSError:
                pass

    def say(self, content):
        self.messages.append({"role": "user", "content": content})
        self.record("user", content)

    def ask(self):
        print(f"\n{C.d}--- debux ---{C.r}")
        # A 27B model takes the better part of a minute to answer. Without a marker the silence
        # is indistinguishable from a hang, which is what it gets reported as.
        state = {"first": True}

        def emit(tok):
            if state["first"]:
                sys.stdout.write("\r" + " " * 30 + "\r")
                state["first"] = False
            sys.stdout.write(tok); sys.stdout.flush()

        sys.stdout.write(f"{C.d}thinking...{C.r}"); sys.stdout.flush()
        try:
            reply = self.backend.chat(protocol.windowed(self.messages), on_token=emit)
        except backends.BackendError as e:
            print(f"\n{C.rd}{e}{C.r}")
            return None
        except KeyboardInterrupt:
            print(f"\n{C.d}interrupted{C.r}")
            return None
        print()
        self.messages.append({"role": "assistant", "content": reply})
        kind, block, _ = protocol.split(reply)
        self.record("assistant", reply, kind)
        return kind, block, reply

    # ---------------------------------------------------------------- input

    @staticmethod
    def block(prompt):
        """Read pasted output. A paste submits itself; typed input ends with '.' or Ctrl-D."""
        return reader.read_block(prompt)

    # ---------------------------------------------------------------- commands

    def cmd_connect(self, arg):
        opts = {"url": None, "token": None, "status": False, "disconnect": False}
        try:
            parts = shlex.split(arg or "")
        except ValueError:
            parts = (arg or "").split()
        i = 0
        while i < len(parts):
            p = parts[i]
            low = p.lower()
            if low in ("disconnect", "off", "clear", "--disconnect"):
                opts["disconnect"] = True
            elif low in ("status", "--status"):
                opts["status"] = True
            elif p in ("--token", "-t") and i + 1 < len(parts):
                opts["token"] = parts[i + 1]; i += 1
            elif p.startswith("--token="):
                opts["token"] = p.split("=", 1)[1]
            elif opts["url"] is None:
                opts["url"] = p
            i += 1

        if opts["disconnect"]:
            self.backend = self.local
            print(f"{C.gr}disconnected; using {self.backend.describe()}{C.r}")
            return
        if opts["status"] or not opts["url"]:
            ok, msg = backends.status(self.backend)
            colour = C.gr if ok else C.rd
            print(f"  backend  {self.backend.describe()}")
            print(f"  status   {colour}{msg}{C.r}")
            if not opts["url"] and not opts["status"]:
                print(f"{C.d}  usage: /connect <url> [--token t] | /connect status | "
                      f"/connect disconnect{C.r}")
            return

        url = opts["url"].strip().rstrip("/")
        if "://" not in url:
            local = url.startswith(("localhost", "127.", "0.0.0.0", "[::1]"))
            url = ("http://" if local else "https://") + url
        if not url.endswith("/v1"):
            url += "/v1"
        token = opts["token"] or os.environ.get("DEBUX_TOKEN")
        cand = backends.OpenAICompat(self.backend.model, url, token, label=url)
        ok, msg = backends.status(cand)
        if not ok:
            print(f"{C.rd}not connected: {msg}{C.r}")
            return
        self.backend = cand
        print(f"{C.gr}connected: {msg}{C.r}")
        conf = load_conf()
        conf["last_connect"] = url
        if token:
            conf["token"] = token
        save_conf(conf)

    def cmd_search(self, arg):
        if not arg:
            print(f"{C.d}usage: /search <query>{C.r}")
            return
        print(f"{C.ma}searching:{C.r} {arg}")
        res = tools.search(arg)
        print(f"{C.d}{wrap(res)}{C.r}")
        self.say(f"Search results for '{arg}':\n\n{res}\n\n"
                 "Use these only if they settle it; if they do not, say so.")

    def cmd_fetch(self, arg):
        if not arg:
            print(f"{C.d}usage: /fetch <url>{C.r}")
            return
        print(f"{C.ma}fetching:{C.r} {arg}")
        try:
            body = tools.fetch(arg)
        except tools.FetchError as e:
            print(f"{C.rd}{e}{C.r}")
            return
        print(f"{C.d}{wrap(body[:600])}{C.r}\n{C.d}  ({len(body)} chars added){C.r}")
        self.say(f"Contents of {arg}:\n\n{body}")

    def cmd_upload(self, arg):
        if not arg:
            print(f"{C.d}usage: /upload <path>{C.r}")
            return
        for path in shlex.split(arg):
            try:
                body = tools.read_file(path)
            except tools.FetchError as e:
                print(f"{C.rd}{e}{C.r}")
                continue
            print(f"{C.gr}added {path}{C.r} {C.d}({len(body)} chars){C.r}")
            self.say(f"File the user attached:\n\n{body}")

    def cmd_save(self, arg):
        path = (arg or "").strip() or f"debux-{time.strftime('%Y%m%d-%H%M%S')}.md"
        try:
            with open(path, "w") as fh:
                fh.write("# Debux session\n\n")
                for e in self.log:
                    who = "You" if e["role"] == "user" else "Debux"
                    tag = f" ({e['kind']})" if e.get("kind") else ""
                    fh.write(f"## {who}{tag}\n\n```\n{e['content']}\n```\n\n")
            print(f"{C.gr}saved {path}{C.r}")
        except OSError as e:
            print(f"{C.rd}could not save: {e}{C.r}")

    def handle_command(self, line):
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            for c, d in HELP:
                print(f"  {C.cy}{c:<30}{C.r} {C.d}{d}{C.r}")
        elif cmd == "/connect":
            self.cmd_connect(arg)
        elif cmd == "/search":
            self.cmd_search(arg)
        elif cmd == "/fetch":
            self.cmd_fetch(arg)
        elif cmd in ("/upload", "/file", "/add"):
            self.cmd_upload(arg)
        elif cmd == "/paste":
            out = self.block(f"{C.ye}paste output, end with a lone '.':{C.r}")
            if out:
                self.say(out)
                self.turn()
        elif cmd == "/model":
            if arg:
                self.backend.model = arg
                print(f"{C.gr}model set to {arg}{C.r}")
            else:
                print(f"  {self.backend.describe()}")
        elif cmd == "/reset":
            self.messages = [{"role": "system", "content": self.system}]
            print(f"{C.gr}conversation cleared{C.r}")
        elif cmd == "/save":
            self.cmd_save(arg)
        else:
            print(f"{C.d}unknown command; /help for the list{C.r}")
        return True

    # ---------------------------------------------------------------- the loop

    def auto_fetch(self, url):
        print(f"\n{C.ma}fetching:{C.r} {url}")
        try:
            body = tools.fetch(url)
        except tools.FetchError as e:
            print(f"{C.rd}{e}{C.r}")
            self.say(f"Fetching {url} failed: {e}. Continue without it, or say what you need.")
            return
        print(f"{C.d}{wrap(body[:400])}{C.r}\n{C.d}  ({len(body)} chars added){C.r}")
        self.say(f"Contents of {url}:\n\n{body}")

    def maybe_assist(self, reply):
        """Run a lookup the model needed but did not ask for.

        It emits SEARCH very rarely, so waiting for the directive means the tool never fires. When
        the text says outright that something depends on a version or vendor fact, searching for it
        is the useful move - and better than letting it guess, which is the failure this whole
        model is built against.
        """
        if self.no_search:
            return False
        found = protocol.urls(reply)
        if found:
            self.auto_fetch(found[0])
            return True
        if not protocol.sounds_unsure(reply):
            return False
        query = self.assist_query(reply)
        if not query:
            return False
        print(f"\n{C.ma}it flagged an unknown - searching:{C.r} {query}")
        res = tools.search(query)
        print(f"{C.d}{wrap(res)}{C.r}")
        self.say(f"You said this depends on something you should not invent, so here are search "
                 f"results for '{query}':\n\n{res}\n\nUse them only if they settle it; if they "
                 f"do not, say so.")
        return True

    def assist_query(self, reply):
        """Build a query from the problem and the reply's own nouns - versions, packages, errors."""
        import re
        first = next((m["content"] for m in self.messages if m["role"] == "user"), "")
        bits = re.findall(r"[A-Za-z][\w.+-]*\d[\w.+-]*|\b[a-z][a-z0-9_-]{3,}\b", reply)
        seen, keep = set(), []
        for b in bits:
            bl = b.lower()
            if bl in seen or bl in ("depends", "version", "release", "documentation", "should",
                                    "which", "would", "there", "their", "because", "confirm"):
                continue
            seen.add(bl); keep.append(b)
            if len(keep) >= 8:
                break
        return " ".join(first.split()[:10] + keep[:8]).strip()

    def turn(self):
        """Drive one exchange, following directives until the model needs the user again."""
        budget = 2   # cap on unrequested tool calls per turn, so it cannot loop on itself
        while True:
            got = self.ask()
            if not got:
                return
            kind, block, _ = got

            if kind is None:
                print(f"\n{C.ye}that reply had no directive.{C.r}")
                more = self.block(f"{C.d}add context, or '.' alone to make it try again:{C.r}")
                self.say(more or "Your reply had no directive. Answer again, ending with exactly "
                                 "one of NEED, DIAGNOSIS, SEARCH or INSUFFICIENT.")
                continue

            if kind == "SEARCH":
                q = protocol.query(block)
                if self.no_search:
                    self.say("Search is disabled in this session. Continue from the evidence you "
                             "have, or say exactly what you would need.")
                    continue
                print(f"\n{C.ma}searching:{C.r} {q}")
                res = tools.search(q)
                print(f"{C.d}{wrap(res)}{C.r}")
                self.say(f"Search results for '{q}':\n\n{res}\n\n"
                         "Use these if they settle it. If they do not, say so rather than "
                         "inferring an answer they do not support.")
                continue

            if kind == "FETCH":
                target = block.split(":", 1)[1].strip().splitlines()[0].strip()
                self.auto_fetch(target)
                continue

            if kind in ("DIAGNOSIS", "INSUFFICIENT", "PROCEDURE"):
                # Even when it commits, it may have leaned on something it should have looked up.
                if self.auto_tools and budget > 0 and self.maybe_assist(reply):
                    budget -= 1
                    continue
                return

            cmds = protocol.commands(block)
            if cmds:
                print(f"\n{C.d}run these and paste what they print:{C.r}")
                for c in cmds:
                    print(f"  {C.cy}${C.r} {c}")
            out = self.block(f"\n{C.ye}paste the output, end with a lone '.':{C.r}")
            if not out:
                return
            self.say(out)


def build_backend(a, conf):
    if a.api_base:
        return backends.OpenAICompat(a.model, a.api_base.rstrip("/"),
                                     a.token or conf.get("token"))
    return backends.Ollama(a.model, a.host)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="debux", description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("DEBUX_MODEL", "debux"))
    ap.add_argument("--host", default="http://127.0.0.1:11434", help="ollama host")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL")
    ap.add_argument("--token", default=None)
    ap.add_argument("--system", default=None, help="path to a system prompt")
    ap.add_argument("--transcript", default=None, help="append the session to this jsonl file")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--no-auto-tools", action="store_true",
                    help="only search or fetch when the model explicitly asks")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("problem", nargs="*", help="the problem, stated on the command line")
    a = ap.parse_args(argv)

    if a.no_color or not sys.stdout.isatty():
        C.off()
    conf = load_conf()
    try:
        system = protocol.system_prompt(a.system)
    except OSError as e:
        print(f"cannot read system prompt: {e}", file=sys.stderr)
        return 1

    s = Session(build_backend(a, conf), system, a.transcript, a.no_search,
                auto_tools=not a.no_auto_tools)

    print(f"{C.b}Debux{C.r}  {C.d}{s.backend.describe()}{C.r}")
    print(f"{C.d}Describe the problem. Paste command output when asked, ending with a lone '.'.")
    print(f"Debux never runs anything on your machine. /help for commands.{C.r}\n")

    first = " ".join(a.problem).strip()
    while True:
        if first:
            line, first = first, ""
        else:
            try:
                line = input(f"{C.cy}problem>{C.r} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
        if not line:
            continue
        if line.startswith("/"):
            if not s.handle_command(line):
                return 0
            continue
        s.say(line)
        s.turn()


if __name__ == "__main__":
    sys.exit(main())
