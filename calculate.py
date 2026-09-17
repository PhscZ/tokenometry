#!/usr/bin/env python3
import os
import sys
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

try:
    import tiktoken
except ImportError:
    print("Missing tiktoken. Install with: pip install --upgrade tiktoken")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

SCRIPT_DIR = Path(__file__).resolve().parent

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    ".terraform",
}

IGNORE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".ico",
    ".webp",
    ".tiff",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".pyc",
    ".pyo",
    ".pyd",
    ".class",
    ".jar",
    ".war",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wav",
    ".flac",
    ".ogg",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".lock",
    ".sum",
}

IGNORE_FILES = {
    ".env",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "go.sum",
}


def parse_models_from_env(env_path: Path):
    models = {}

    # dotenv_values reads the file top-to-bottom, preserving order
    env_vars = dotenv_values(env_path)

    pattern = re.compile(
        r"^MODEL_([A-Z0-9_]+)_(TOKENIZER|INPUT|OUTPUT|CACHE|NAME)$"
    )

    for key, value in env_vars.items():
        if not value:
            continue

        match = pattern.match(key)
        if not match:
            continue

        model_id = match.group(1)
        prop = match.group(2).lower()

        if model_id not in models:
            models[model_id] = {
                "id": model_id,
                "display_name": model_id.replace("_", " ").title(),
                "tokenizer": None,
                "input": 0.0,
                "output": 0.0,
                "cache": 0.0,
            }

        if prop == "tokenizer":
            models[model_id]["tokenizer"] = value
        elif prop == "name":
            models[model_id]["display_name"] = value
        else:
            try:
                models[model_id][prop] = float(value)
            except ValueError:
                pass

    # Keep only models that have a tokenizer
    # Python dictionaries preserve insertion order
    return {k: v for k, v in models.items() if v["tokenizer"]}


def load_tokenizers(models):
    tokenizers = {}
    hf_token = os.getenv("HF_TOKEN") or None

    # Load each unique tokenizer only once
    unique_tokenizers = set(m["tokenizer"] for m in models.values())

    for tk_name in unique_tokenizers:
        print(f"✓ Loading tokenizer: {tk_name}")

        try:
            if tk_name in ["o200k_base", "cl100k_base", "p50k_base"]:
                tokenizers[tk_name] = tiktoken.get_encoding(tk_name)
            else:
                if AutoTokenizer is None:
                    print(f"✗ transformers is required for {tk_name}")
                    continue

                tokenizers[tk_name] = AutoTokenizer.from_pretrained(
                    tk_name,
                    token=hf_token,
                    trust_remote_code=True,
                )
        except Exception as exc:
            print(f"✗ Failed to load {tk_name}: {exc}")

    return tokenizers


def scan_folder(folder_path: str, tokenizers: dict):
    counts = {tk_name: 0 for tk_name in tokenizers}
    files_scanned = 0
    unreadable_files = 0

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [
            d
            for d in dirs
            if d not in IGNORE_DIRS and not d.startswith(".")
        ]

        for filename in files:
            file_path = Path(root) / filename

            if (
                filename.startswith(".")
                or file_path.name in IGNORE_FILES
                or file_path.suffix.lower() in IGNORE_EXTS
            ):
                continue

            try:
                content = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                for tk_name, tokenizer in tokenizers.items():
                    if isinstance(tokenizer, tiktoken.Encoding):
                        counts[tk_name] += len(
                            tokenizer.encode_ordinary(content)
                        )
                    else:
                        counts[tk_name] += len(
                            tokenizer.encode(
                                content,
                                add_special_tokens=False,
                            )
                        )

                files_scanned += 1

            except Exception:
                unreadable_files += 1
                continue

    return counts, files_scanned, unreadable_files


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens and estimate costs using a .env model configuration."
    )

    parser.add_argument(
        "folder",
        help="Path to the project folder to scan",
    )

    parser.add_argument(
        "--env",
        "-e",
        dest="env_file",
        default=None,
        help=(
            "Path to the env file to load. "
            "Defaults to .env next to token_counter.py."
        ),
    )

    args = parser.parse_args()

    # Resolve env file path
    if args.env_file:
        env_path = Path(args.env_file).expanduser().resolve()
    else:
        env_path = SCRIPT_DIR / ".env"

    if not env_path.is_file():
        sys.exit(f"Env file not found: {env_path}")

    # Load env file into os.environ, useful for HF_TOKEN
    load_dotenv(env_path)

    all_models = parse_models_from_env(env_path)

    if not all_models:
        sys.exit(
            f"No models found in {env_path}. "
            "Please check your MODEL_ variables."
        )

    tokenizers = load_tokenizers(all_models)

    if not tokenizers:
        sys.exit("No tokenizers were loaded successfully.")

    print(f"\nUsing env file: {env_path}")
    print(f"Scanning folder: {args.folder}")
    print("-" * 70)

    counts, files_scanned, unreadable_files = scan_folder(
        args.folder,
        tokenizers,
    )

    print("\n--- Token Count & Cost Estimation (per 1M Tokens) ---")
    print(
        f"Folder: {args.folder} | "
        f"Scanned: {files_scanned} | "
        f"Skipped: {unreadable_files}"
    )
    print()

    header = (
        f"{'Model':<35} | "
        f"{'Tokens':>12} | "
        f"{'Std Input':>12} | "
        f"{'Cache Input':>12} | "
        f"{'Output':>12} | "
        f"{'TOTAL':>12}"
    )

    print(header)
    print("-" * 115)

    # Output respects the order of the selected env file
    for model_id, info in all_models.items():
        tk_name = info["tokenizer"]
        tokens = counts.get(tk_name, 0)

        if tk_name not in counts:
            continue

        std_cost = (tokens / 1_000_000) * info["input"]
        cache_cost = (tokens / 1_000_000) * info["cache"]
        output_cost = (tokens / 1_000_000) * info["output"]
        total_cost = std_cost + cache_cost + output_cost

        print(
            f"{info['display_name']:<35} | "
            f"{tokens:>12,} | "
            f"${std_cost:>11.2f} | "
            f"${cache_cost:>11.2f} | "
            f"${output_cost:>11.2f} | "
            f"${total_cost:>11.2f}"
        )


if __name__ == "__main__":
    main()
