"""
Module: llm_cleaner.py
Purpose: Clean pre-processed medical text using an LLM to remove boilerplate,
         correct line breaks, and structure it while preserving all medical facts.

Supports multiple provider API keys (comma-separated in .env).
When one key hits its quota (429), it automatically rotates to the next.
Only stops if ALL keys are exhausted.

Features:
    - Atomic file writes (no half-written files on crash)
    - Resume / checkpointing (skips already-processed files)
    - Rate limiting (configurable delay between API calls)
    - Output validation (detects potential data loss)
    - Temperature = 0.0 for deterministic medical output
    - Context length warning

Usage:
    python -m src.ingestion.llm_cleaner
    python -m src.ingestion.llm_cleaner --processed-dir data/processed --cleaned-dir data/cleaned
    python -m src.ingestion.llm_cleaner --dry-run
    python -m src.ingestion.llm_cleaner --delay 1.5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError

from src.llm.client import LLMClient
from src.llm import config as llm_config
from src.config import AppSettings, get_settings
from src.utils.helpers import setup_logging

log = setup_logging("llm_cleaner")

# ── Defaults ─────────────────────────────────────────────────────────
_DEFAULT_MODEL = llm_config.DEFAULT_MODEL
_DEFAULT_PROMPT_FILE = "prompts/cleaning/medical_clean.txt"
_DEFAULT_DELAY = 0.5  # seconds between API calls
_MAX_CONTEXT_CHARS = 500_000  # rough safety limit (~125k tokens for 4-char avg)


# ── Pydantic schema for structured output ────────────────────────────
class MedicalCleanedData(BaseModel):
    cleaned_content: str = Field(
        description=(
            "The final cleaned medical text. All extra whitespaces, incorrect "
            "paragraph breaks, and non-medical metadata are removed. The core "
            "medical facts, terminology, numbers, and warnings are kept completely "
            "intact without modification."
        )
    )


# ── API Key Management ───────────────────────────────────────────────
def load_api_keys(model: str, settings: AppSettings | None = None) -> list[str]:
    """Parse comma-separated API keys for the selected LiteLLM provider."""
    provider = model.split("/", 1)[0].strip().lower()
    key_name = {
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    if key_name is None:
        return []

    app_settings = settings or get_settings()
    secret_field = {
        "GEMINI_API_KEY": app_settings.secrets.gemini_api_key,
        "GROQ_API_KEY": app_settings.secrets.groq_api_key,
        "OPENAI_API_KEY": app_settings.secrets.openai_api_key,
        "OPENROUTER_API_KEY": app_settings.secrets.openrouter_api_key,
        "ANTHROPIC_API_KEY": app_settings.secrets.anthropic_api_key,
    }[key_name]
    raw = secret_field
    return [key.strip() for key in raw.split(",") if key.strip()]


# ── Prompt Loader ────────────────────────────────────────────────────
def load_prompt(prompt_path: Path) -> str:
    """Read a prompt template from a text file.

    Args:
        prompt_path: Path to the prompt ``.txt`` file.

    Returns:
        Prompt text content.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as fh:
        return fh.read().strip()





# ── Output Validation ────────────────────────────────────────────────
def validate_output(original: str, cleaned: str, file_name: str) -> str:
    """Validate cleaned output to guard against catastrophic data loss.

    If the cleaned text is suspiciously short compared to the original,
    logs a warning and returns the original text instead.

    Args:
        original: Original raw text.
        cleaned: LLM-cleaned text.
        file_name: Identifier for logging.

    Returns:
        The cleaned text if validation passes, otherwise the original text.
    """
    orig_len = len(original.strip())
    clean_len = len(cleaned.strip())

    if orig_len == 0:
        return cleaned

    ratio = clean_len / orig_len
    if ratio < 0.5:
        log.warning(
            f"[{file_name}] Potential data loss detected: "
            f"cleaned length is {ratio:.1%} of original ({clean_len}/{orig_len}). "
            f"Keeping original text."
        )
        return original

    return cleaned


# ── Atomic Writer ────────────────────────────────────────────────────
def write_json_atomic(dest_path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically using a temp file + rename to avoid half-written files.

    Args:
        dest_path: Final destination path.
        data: Dictionary to serialize as JSON.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)
    tmp_path.replace(dest_path)


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Clean medical text using the configured LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed source subdirectories.",
    )
    p.add_argument(
        "--cleaned-dir",
        default="data/cleaned",
        help="Output directory for LLM-cleaned files.",
    )
    p.add_argument(
        "--prompt-file",
        default=_DEFAULT_PROMPT_FILE,
        help="Path to the system prompt .txt file.",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="Override the centralized LLM model for this cleaning run.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=_DEFAULT_DELAY,
        help="Delay in seconds between consecutive API calls.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to process without calling the API.",
    )
    return p.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — clean medical text with Gemini."""
    args = parse_args(argv)
    settings = get_settings()
    root = settings.project_root

    model = llm_config.normalize_model(args.model)
    api_keys = load_api_keys(model, settings)
    provider = model.split("/", 1)[0].strip().lower()
    if not api_keys and not args.dry_run and provider not in {"ollama", "local"}:
        log.error("No API key found for provider '%s'. Add it to .env or export it.", provider)
        sys.exit(1)

    log.info("Loaded %d API key(s) for provider %s.", len(api_keys), provider)

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir

    cleaned_dir = Path(args.cleaned_dir)
    if not cleaned_dir.is_absolute():
        cleaned_dir = root / cleaned_dir

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = root / prompt_path

    if not processed_dir.is_dir():
        log.error(f"Processed data directory not found: {processed_dir}")
        sys.exit(1)

    # Load system prompt
    system_prompt = load_prompt(prompt_path)
    log.info(f"Loaded system prompt from: {prompt_path}")

    if model != args.model.strip():
        log.info("Normalized model id: %s → %s", args.model, model)

    # Initialize the LLMClient with all API keys for automatic rotation
    client = LLMClient(
        default_model=model,
        temperature=llm_config.TEMPERATURE,
        api_keys=api_keys,
        settings=settings,
    )

    total_files = 0
    skipped_files = 0
    processed_files = 0
    failed_files = 0

    log.info(
        "Starting LLM-based cleaning with %s (delay=%ss, temperature=%s)...",
        model,
        args.delay,
        llm_config.TEMPERATURE,
    )

    try:
        for source_path in sorted(processed_dir.iterdir()):
            if not source_path.is_dir():
                continue

            # Standardize source folder name (e.g. "mayo clinic" → "mayo")
            source_name = source_path.name.lower()
            if "mayo" in source_name:
                source_name = "mayo"

            dest_source_dir = cleaned_dir / source_name
            dest_source_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(source_path.glob("*.json"))
            log.info(f"Processing '{source_name}' ({len(files)} files)...")

            if args.dry_run:
                for fp in files:
                    log.info(f"[DRY RUN] Would clean: {fp.name}")
                continue

            for file_path in tqdm(files, desc=f"Cleaning {source_name}"):
                total_files += 1
                dest_file = dest_source_dir / file_path.name.lower()

                # ── Checkpointing: skip already processed ───────────────
                if dest_file.exists():
                    skipped_files += 1
                    continue

                with open(file_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                content = data.get("content", "").strip()
                original_content = content

                if not content:
                    cleaned_content = ""
                else:
                    # Warn if content looks too large for context window
                    if len(content) > _MAX_CONTEXT_CHARS:
                        log.warning(
                            f"[{file_path.name}] Content length ({len(content)} chars) "
                            f"exceeds safe context limit. May fail or truncate."
                        )

                    try:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content}
                        ]
                        response = client.complete(
                            messages=messages,
                            response_format=MedicalCleanedData,
                        )
                        # Extract parsed content from standard LiteLLM ModelResponse choices
                        response_text = response.choices[0].message.content
                        result = MedicalCleanedData.model_validate_json(response_text)
                        
                        cleaned_content = validate_output(
                            original=original_content,
                            cleaned=result.cleaned_content,
                            file_name=file_path.name,
                        )
                        processed_files += 1

                    except (ValidationError, json.JSONDecodeError) as e:
                        log.warning(f"Failed to parse cleaned content for {file_path.name}: {e}")
                        cleaned_content = original_content
                        failed_files += 1

                    except KeyboardInterrupt:
                        log.warning("Interrupted by user. Shutting down gracefully...")
                        return

                    except Exception as e:
                        log.error(f"Failed to clean {file_path.name} via LLM: {e}")
                        cleaned_content = original_content
                        failed_files += 1
                        
                        # Exit early if all keys are rate-limited / exhausted
                        err_str = str(e).lower()
                        if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                            log.error("Stopping pipeline — all API keys exhausted.")
                            return

                cleaned_data = {
                    "title": data.get("title", ""),
                    "url": data.get("url", ""),
                    "source": data.get("source", ""),
                    "content": cleaned_content,
                }

                write_json_atomic(dest_file, cleaned_data)

                # ── Rate limiting ───────────────────────────────────────
                if args.delay > 0:
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        log.warning("Interrupted by user. Shutting down gracefully...")
        return

    log.info(
        "LLM-based text cleaning completed. "
        f"Total: {total_files}, Processed: {processed_files}, "
        f"Skipped: {skipped_files}, Failed: {failed_files}."
    )


if __name__ == "__main__":
    main()
