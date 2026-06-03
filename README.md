# CodePi — Raspberry Pi Local Coding Agent

CodePi is a coding-only AI assistant that runs locally on Raspberry Pi using
llama.cpp through `llama-cpp-python`. It ships in two runtime modes — a kid-
friendly 8-bit web GUI and a pure interactive CLI — and supports multiple GGUF
models for benchmarking and testing.

## Features

- Local inference only, no cloud API
- Two runtime modes:
  - `--mode web` (default) — retro 8-bit pixel-art chat GUI with a reactive
    face that gets happy / thinking / confused / unhappy based on the query
  - `--mode cli` — interactive terminal REPL (pure backend, no Flask)
- Multi-model support: drop any `*.gguf` into `./models/` and it shows up
  automatically in the model dropdown and `:models` command. **No filenames
  are hardcoded** — the list is derived from the directory at runtime.
- Built-in benchmark: run the same prompt across every discovered model and
  compare latency / approx tokens-per-second.
- Conversation memory: the agent remembers the last `MAX_HISTORY_TURNS` of
  the conversation (default 6) so follow-ups like "now optimize that" work.
- Coding-only filtering plus rudeness/jailbreak detection.

## Hardware

Recommended:

- Raspberry Pi 5
- 16GB RAM
- Raspberry Pi OS Lite 64-bit

## Setup on Raspberry Pi

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git python3-venv python3-pip cmake build-essential
```

Clone this repository:

```bash
git clone <your-repo-url>
cd coding-pi-agent
```

Create Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Models

Drop one or more GGUF files into `./models/`. CodePi scans this directory at
startup; any `*.gguf` becomes a selectable model. Any GGUF works — Qwen-Coder,
Gemma / CodeGemma, Phi-3, TinyLlama, DeepSeek-Coder, Llama-3, etc. Example
layout:

```text
models/
  qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
  qwen2.5-coder-7b-instruct-q4_k_m.gguf
  gemma-2-2b-it-q4_k_m.gguf
  codegemma-1.1-2b-q4_k_m.gguf
  phi-3-mini-4k-instruct-q4.gguf
  tinyllama-1.1b-chat.q4_k_m.gguf
```

Override the scan location via env var if you want:

```bash
export MODELS_DIR=/mnt/usb/models
export DEFAULT_MODEL_ID=qwen2.5-coder-1.5b-instruct-q4_k_m
```

`DEFAULT_MODEL_ID` is the lowercased filename without the `.gguf` extension.

## Run — Web GUI mode (default)

```bash
python app.py
# or explicitly
python app.py --mode web --port 5000
```

Then from your laptop browser:

```text
http://<raspberry-pi-ip>:5000
```

The face on the side reacts in real time:

| State    | When                                          |
| -------- | --------------------------------------------- |
| happy    | idle, or a coding answer was just produced    |
| thinking | inference is running                          |
| confused | the question is not about coding              |
| unhappy  | the message was rude or a jailbreak attempt   |

The right-side `MEMORY` bar shows how many conversation turns the agent is
currently keeping in context. Click `CLEAR` to wipe both the chat and memory.

There's also a `BENCH ALL` button that runs the current prompt across every
discovered model and shows side-by-side timing.

## Run — CLI / pure-backend mode

```bash
python app.py --mode cli
# optionally pick the starting model
python app.py --mode cli --model qwen2.5-coder-1.5b-instruct-q4_k_m
```

Inside the REPL:

```text
:help            show commands
:models          list discovered GGUFs
:use <model_id>  switch active model
:bench <prompt>  benchmark across every discovered model
:reset           clear conversation memory (history)
:history         show how many turns are remembered
:clear           clear screen
:quit            exit
```

### Conversation memory

The agent remembers prior turns so follow-ups work naturally:

```text
you > Explain bubble sort in Python.
CodePi > [...]
you > Now make it iterative.
CodePi > [continues from previous answer]
```

Set the window with `MAX_HISTORY_TURNS` (default `6`). One turn = one user
message + one assistant reply. Rude or non-coding messages are filtered before
the model is called and do NOT enter history.

## HTTP API (web mode)

| Method | Path             | Body                                       |
| ------ | ---------------- | ------------------------------------------ |
| GET    | `/api/health`    | —                                          |
| GET    | `/api/models`    | —                                          |
| POST   | `/api/chat`      | `{ "message": "...", "model_id": "..." }`  |
| POST   | `/api/benchmark` | `{ "message": "...", "model_ids": [...] }` |

`/api/chat` returns `{ reply, mode, face_state, model_id, elapsed_ms }`.
`face_state` is one of `happy | thinking | confused | unhappy` and drives the
pixel face in the GUI.

## Example questions

```text
Solve two sum in C++.
Debug this segmentation fault.
Explain this Python function.
Generate unit tests for binary search.
Optimize this O(n^2) algorithm.
```

## Model notes

Recommended coding model:

```text
Qwen2.5-Coder-7B-Instruct GGUF, Q4_K_M
```

For faster first test, use a smaller model:

```text
Qwen2.5-Coder-1.5B-Instruct GGUF
```
