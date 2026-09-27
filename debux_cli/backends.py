"""Model backends: Ollama, any OpenAI-compatible endpoint, and a remote one via /connect.

All three stream, because a 27B model on a laptop produces tokens slowly enough that a silent
wait feels like a hang.

Thinking is switched off on every backend. This base has a thinking mode, and with it enabled the
model spends its budget narrating the output format instead of producing it - the training corpus
contains no think blocks, so reasoning belongs in the visible answer.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


class BackendError(Exception):
    pass


class Ollama:
    kind = "ollama"

    def __init__(self, model, host="http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def describe(self):
        return f"{self.model} via ollama at {self.host}"

    def chat(self, messages, on_token=None, temperature=0.2):
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": True,
            "think": False, "options": {"temperature": temperature},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        parts = []
        try:
            with urllib.request.urlopen(req) as r:
                for line in r:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    chunk = o.get("message", {}).get("content")
                    if chunk:
                        parts.append(chunk)
                        if on_token:
                            on_token(chunk)
                    if o.get("done"):
                        break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise BackendError(
                    f"ollama is running but has no model called '{self.model}'.\n"
                    f"  pull one:  ollama pull {self.model}\n"
                    f"  or list what it has:  ollama list\n"
                    f"  or point at local weights instead:  --serve /path/to/model")
            raise BackendError(f"ollama returned HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise BackendError(f"cannot reach ollama at {self.host}: {e.reason}")
        return "".join(parts)


class OpenAICompat:
    """llama-server, vLLM, an mlx_lm server, or a remote box reached with /connect."""
    kind = "openai"

    def __init__(self, model, base, token=None, label=None):
        self.model = model
        self.base = base.rstrip("/")
        self.token = token
        self.label = label

    def describe(self):
        where = self.label or self.base
        return f"{self.model} via {where}" + (" (token set)" if self.token else "")

    def chat(self, messages, on_token=None, temperature=0.2):
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": True,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body, headers=headers)
        parts = []
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    chunk = delta.get("content")
                    if chunk:
                        parts.append(chunk)
                        if on_token:
                            on_token(chunk)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace") if e.fp else ""
            raise BackendError(f"HTTP {e.code} from {self.base}: {detail}")
        except urllib.error.URLError as e:
            raise BackendError(f"cannot reach {self.base}: {e.reason}")
        return "".join(parts)


def status(backend):
    """A short reachability check, so /connect can report rather than fail on the next prompt."""
    try:
        if backend.kind == "ollama":
            with urllib.request.urlopen(f"{backend.host}/api/tags", timeout=10) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
            return True, f"{len(names)} model(s) available" + (
                "" if backend.model in names else f"; note: {backend.model} is not among them")
        req = urllib.request.Request(f"{backend.base}/models")
        if backend.token:
            req.add_header("Authorization", f"Bearer {backend.token}")
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data", [])]
        return True, f"reachable, {len(ids)} model(s): {', '.join(str(i) for i in ids[:3])}"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"HTTP {e.code}: authentication rejected - check --token"
        return False, f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


class LocalServer:
    """Runs `mlx_lm server` against local weights so the CLI works without a separate terminal.

    Loading a 27B model takes the better part of a minute, so the wait is reported rather than
    hidden. If something is already serving on the port it is reused instead - starting a second
    copy would try to hold another ~17 GB and fail badly on a 32 GB machine.
    """

    def __init__(self, model_path, port=8080):
        self.model_path = os.path.expanduser(model_path)
        self.port = port
        self.proc = None
        self.base = f"http://127.0.0.1:{port}/v1"

    def already_running(self):
        try:
            with urllib.request.urlopen(f"{self.base}/models", timeout=3) as r:
                served = [m.get("id") for m in json.load(r).get("data", [])]
            return True, served
        except Exception:
            return False, []

    def start(self, log_path=None, timeout=300, on_wait=None):
        up, served = self.already_running()
        if up:
            return "reused", served

        if not os.path.isdir(self.model_path):
            raise BackendError(f"no model directory at {self.model_path}")

        log = open(log_path or os.devnull, "a")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm", "server",
             "--model", self.model_path, "--port", str(self.port)],
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)   # survives Ctrl-C aimed at the CLI

        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise BackendError(
                    f"the model server exited immediately (see {log_path or 'its output'})")
            up, served = self.already_running()
            if up:
                return "started", served
            if on_wait:
                on_wait(time.time() - start)
            time.sleep(2)
        self.stop()
        raise BackendError(f"model server did not come up within {timeout}s")

    def stop(self):
        """Only ever stops a server this process started."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
