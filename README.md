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
- **Query routing (RouteLLM-style)**: a BERT embedder maps each query to the
  nearest LeetCode problems in a RAG index, infers a difficulty rating, and
  routes the query to the model whose ELO best matches. Hard problems go to
  the bigger model; easy ones go to the small fast one. Falls back to a
  keyword heuristic when BERT/RAG aren't set up.
- ELO learning: thumbs up / down on any reply runs one standard ELO update
  (K=32). State persists in `models/elo_state.json`.
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
:use <model_id>  switch active model (turns auto routing OFF)
:auto on|off     turn router on/off
:route <prompt>  dry-run the router; show chosen model + difficulty
:elo             show ELO for every discovered model
:vote up|down    vote on the last reply
:bench <prompt>  benchmark across every discovered model
:reset           clear conversation memory (history)
:history         show how many turns are remembered
:clear           clear screen
:quit            exit
```

### Query routing (RouteLLM-style)

CodePi can auto-route each query to the model whose ELO best matches the
predicted difficulty. The pipeline:

```text
user query
  -> BERT embed (sentence-transformers MiniLM-L6-v2 by default)
  -> KNN against precomputed LeetCode embeddings (from zerotrac/ratings.txt)
  -> weighted-average neighbor ratings = predicted LC rating
  -> linear map -> target model ELO (range 950..1450)
  -> pick available GGUF whose ELO is closest
```

Model ELOs are seeded from **LMSYS Chatbot Arena (Coding category)
approximations** keyed on filename family (e.g. `qwen2.5-coder-7b` → 1185).
Anything in `models/elo_state.json` overrides the seed. Thumbs vote (`/api/vote`
or the GUI buttons) runs one ELO update (K=32) so ratings adapt to your
actual Pi setup.

**One-time setup:**

1. Download `ratings.txt` from
   <https://github.com/zerotrac/leetcode_problem_rating/blob/main/ratings.txt>
2. Build the index (this also fetches the BERT weights into the HF cache
   on first run):

   ```bash
   pip install sentence-transformers numpy
   python scripts/setup_rag.py ./ratings.txt
   ```

   Outputs `data/leetcode_embeddings.npy` and `data/leetcode_meta.json`.

**Using it:**

- Web GUI: leave the dropdown on `AUTO (route by difficulty)`. The bot
  reply shows a routing badge with predicted LC rating, target ELO, and
  the chosen model, plus `+1` / `-1` vote buttons.
- CLI: `:auto on` (default), `:route <prompt>` for a dry-run, `:vote up|down`
  to update the last model's ELO, `:elo` to dump every model's current ELO.
- API: send `model_id: "auto"` in `/api/chat` to enable routing.

Skip the setup and the router gracefully falls back to a keyword heuristic
— the agent still works, routing is just coarser.

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

| Method | Path             | Body                                                                 |
| ------ | ---------------- | -------------------------------------------------------------------- |
| GET    | `/api/health`    | —                                                                    |
| GET    | `/api/models`    | —                                                                    |
| GET    | `/api/elo`       | —                                                                    |
| POST   | `/api/chat`      | `{ "message": "...", "model_id": "auto" \| "<id>", "history": [] }` |
| POST   | `/api/route`     | `{ "message": "..." }`  (dry-run, returns chosen model + difficulty) |
| POST   | `/api/vote`      | `{ "model_id": "...", "vote": "up" \| "down", "target_elo": 1200 }` |
| POST   | `/api/benchmark` | `{ "message": "...", "model_ids": [...] }`                           |

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
