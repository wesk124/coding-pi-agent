# CodePi — Raspberry Pi Local Coding Agent

CodePi is a coding-only AI assistant that runs locally on Raspberry Pi using llama.cpp through `llama-cpp-python`.

## Features

- Local inference only
- No cloud API
- Coding-only query filtering
- Agent modes: debug, explain, optimize, generate tests, solve coding problems
- Local web UI with Flask

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

Create model folder:

```bash
mkdir -p models
```

Copy a GGUF model to:

```text
models/model.gguf
```

Run:

```bash
python app.py
```

Open from your laptop browser:

```text
http://<raspberry-pi-ip>:5000
```

## Example Questions

```text
Solve two sum in C++.
Debug this segmentation fault.
Explain this Python function.
Generate unit tests for binary search.
Optimize this O(n^2) algorithm.
```

## Model Notes

Recommended coding model:

```text
Qwen2.5-Coder-7B-Instruct GGUF, Q4_K_M
```

For faster first test, use a smaller model:

```text
Qwen2.5-Coder-1.5B-Instruct GGUF
```
