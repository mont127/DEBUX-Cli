# Debux CLI

A terminal client for [Debux](https://huggingface.co/MK4-Research/Debux), a debugging model for
Linux and DevOps problems.

You have a broken server. You describe it, Debux tells you what to run, you paste back what it
printed, and it reasons about what that output rules in and out until it can name a cause or say
honestly that it cannot.

**Debux never runs anything on your machine.** You are the hands, it is the head. A debugging
assistant with SSH on a broken production box has a blast radius; this one has none, and you see
every command before it runs.

## Install

```
git clone <this repo> && cd debux-cli
pip install -r requirements.txt
./debux
```

It talks to Ollama by default. For a local MLX or llama.cpp server, or a remote box:

```
./debux --api-base http://127.0.0.1:8080/v1 --model debux
./debux "nginx is returning 502 since this morning"
```

## How a session goes

```
problem> disk writes are failing but df -h shows 20G free

--- debux ---
REASONING:
- ENOSPC with free blocks usually means the filesystem ran out of inodes, not bytes
- df -h reports blocks only, so it can look healthy while the inode table is full
NEED:
  df -i /

run these and paste what they print:
  $ df -i /

paste the output, end with a lone '.':
Filesystem      Inodes   IUsed  IFree IUse% Mounted on
/dev/sda1      2621440 2621440      0  100% /
.
```

Every reply is reasoning followed by exactly one directive, and the CLI acts on it:

| directive | what the CLI does |
|---|---|
| `NEED` | lists the commands, waits for you to paste the output |
| `DIAGNOSIS` | prints the cause and the fix, and hands the prompt back |
| `SEARCH` | runs the search, feeds the results back, continues |
| `INSUFFICIENT` | prints what is missing and what would settle it |

`NEED` and `INSUFFICIENT` are the distinction that matters. `NEED` means the answer is on the
machine and a command will fetch it. `INSUFFICIENT` means it is not - the record was never written,
it was rotated away, it lives with a vendor or a hypervisor, or it needs measurement over time that
nobody started. Being told to keep looking at a machine that does not hold the answer wastes an
outage.

## Commands

| command | |
|---|---|
| `/connect <url> [--token t]` | route the model to a remote OpenAI-compatible server |
| `/connect status` | check the connected server |
| `/connect disconnect` | go back to the local backend |
| `/search <query>` | search the web yourself and add the results |
| `/fetch <url>` | fetch a page and add it as evidence |
| `/upload <path>` | add a file - a log, a unit file, a config |
| `/paste` | paste output without being asked |
| `/model <name>` | switch model |
| `/reset` | start a new problem |
| `/save [path]` | write the transcript to markdown |

`/upload` tails long files, so pointing it at a 200 MB log gives the model the end of it, which is
where the failure usually is. Binary files are refused rather than fed in as noise.

### /connect

```
/connect https://box.example.com --token secret
/connect status
/connect disconnect
```

`/v1` is appended if you leave it off, and `http` is assumed only for localhost. The token is also
read from `DEBUX_TOKEN`. It is stored in `~/.debux/config.json` with mode 0600.

## Fetching is guarded

The model chooses the URLs, and this CLI usually runs inside the network being debugged. A model
talked into fetching `http://169.254.169.254/` would be reading cloud credentials, so `/fetch`:

- allows only `http` and `https`
- refuses URLs carrying credentials
- resolves the host and refuses any private, loopback, link-local, reserved or multicast address,
  including IPv4-mapped IPv6 forms
- re-validates every redirect hop
- pins the connection to the address that passed validation, while keeping the hostname for SNI and
  certificate verification, so a DNS rebind between check and connect cannot redirect the fetch
  inward

A failed search is reported to the model as a failure rather than as an empty result, because a
model that reads "no results" as "no such thing exists" becomes a confident source of wrong
statements.

## Two things the client must get right

**The system prompt.** `system_prompt.txt` ships with the client and the model was trained against
it exactly. Replace it and the directive contract stops being reliable.

**History windowing.** The model was trained on the system prompt, the original symptom, and the
last message - never the full transcript. `protocol.windowed()` reproduces that. Sending the whole
conversation is a context shape the model has never seen.

Thinking mode is switched off on every backend. This base has one, and with it enabled the model
spends its budget narrating the output format instead of producing it.

## Limits

Debux is a way to be systematic about evidence under pressure, not a replacement for someone who
knows the system, and it can be wrong. It covers Linux server work: web servers, systemd, disks and
filesystems, memory, networking, containers, application runtimes, authentication, HTTP and TLS.
Not Windows, not macOS, not embedded.
