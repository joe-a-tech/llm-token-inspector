# Token Inspector

A simple CLI tool for exploring how Large Language Models (LLMs) see text — as **tokens**, not words.

This tool helps you:
- Visualise tokenisation
- Understand how small input changes affect token counts
- See how whitespace and punctuation impact tokens
- Compare two inputs side-by-side
- Estimate token-based API cost

---

## Why this exists

LLMs don’t process text like humans. They operate on **tokens** — chunks of text that may not align with words.

Understanding tokens is fundamental to:
- Prompt engineering
- Context window limits
- Cost optimisation
- Debugging LLM behaviour

This tool makes that visible.

---

## Features

- Token breakdown (ID, decoded value, length)
- Visible whitespace (`␠`, `↵`, etc.)
- Stats (token count, averages, longest tokens)
- Compare two inputs (see exact differences)
- Optional cost estimation

---

## Installation

Clone the repo:

```bash
git clone https://github.com/your-username/token-inspector.git
cd token-inspector
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tiktoken rich
```

---

## Usage

Inspect a single input:

```bash
python token_inspector.py "Some default text here"
```

Compare two inputs:

```bash
python token_inspector.py "hello" " hello"
```

Estimate token cost:

```bash
python token_inspector.py --price-per-1k 0.005 "Your prompt here"
```

Change encoding (optional):

```bash
python token_inspector.py --encoding cl100k_base "Test input"
```

---

## Example experiments:

```bash
python token_inspector.py "hello"
python token_inspector.py " hello"
python token_inspector.py "hello "
python token_inspector.py "hello  world"
python token_inspector.py "congratulations"
python token_inspector.py "Hello_world_from_me"
python token_inspector.py "Python,Java,C,Rust,Go"
```

You’ll notice:
- Spaces are part of tokens
- Token counts can change unexpectedly
- Small input differences lead to different token sequences

---

## Key concept

LLMs work like this:

tokens → predict next token → append → repeat

They do not understand text — they predict patterns.

---

## Project structure

token-inspector/
├── token_inspector.py
├── README.md
└── requirements.txt

---

## Future ideas
- Token visualisation UI
- Support for more encodings
- Integration with live LLM APIs
- Prompt optimisation tools

---

## License

MIT







