# Tokenometry

Tokenometry is a local, configurable token calculator for codebases and text projects.

It scans a folder, counts how many tokens your files contain, and estimates how much those tokens would cost across multiple AI models.

Tokenometry does not send your project files to paid LLM APIs. It loads tokenizers locally and counts tokens on your machine.

---

## Features

- Scans an entire folder recursively
- Ignores common build artifacts, dependencies, binaries, and lockfiles
- Supports multiple models in a single run
- Fully configurable through `.env` files
- Supports loading a custom env file with `--env`
- Calculates:
  - total tokens
  - standard input cost
  - cached input cost
  - output cost for the same token amount
  - combined total estimate
- Preserves model order from the env file
- Loads shared tokenizers only once
- Works with:
  - `tiktoken` encodings, such as `o200k_base`
  - Hugging Face tokenizers, such as `deepseek-ai/DeepSeek-V3` or `Qwen/Qwen2.5-7B-Instruct`

---

## What it calculates

For each configured model, Tokenometry outputs:

| Column | Meaning |
|---|---|
| `Tokens` | Number of tokens counted in your project |
| `Std Input` | Cost if all counted tokens are sent as standard input |
| `Cache Input` | Cost if all counted tokens are treated as cached input |
| `Output` | Cost if the model generated the same number of tokens as output |
| `TOTAL` | `Std Input + Cache Input + Output` |

> The `TOTAL` column is a combined estimate. In real usage, a request is usually either standard input, cached input, or output generation, not all three at once.

---

## Requirements

- Python 3.9 or newer recommended
- `pip`
- Internet access on first run, so tokenizer files can be downloaded

---

## Installation

Clone or download the project, then install dependencies:

```bash
pip install -r requirements.txt
```

Create your own .env or rename/edit .env.example

Run with
```bash
python calculate.py /path/to/your/project
```

Run with a different --env
```bash
python calculate.py /path/to/your/project --env .env.example
```
```bash
python calculate.py /path/to/your/project --env config/cheap.env
```
```bash
python calculate.py /path/to/your/project --env "D:\Tools\tokenometry\.env"
```

Windows example
```bash
python calculate.py "D:\Projects\my-game\scripts"
```
```bash
python calculate.py "D:\Projects\my-game\scripts" --env "D:\Tools\tokenometry\.env"
```

To get a Hugging Face token:
1. Go to huggingface.co
2. Log in
3. Open your profile menu
4. Go to Settings
5. Open Access Tokens
6. Create a new token with Read access
7. Paste it into your env file, otherwise keep it empty
```
HF_TOKEN=
```
