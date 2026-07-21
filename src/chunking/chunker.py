"""
Module: chunker.py
Purpose: Split cleaned medical articles and clinic data into indexed chunks
         suitable for RAG vector indexing.

Usage:
    python -m src.chunking.chunker
    python -m src.chunking.chunker --cleaned-dir data/cleaned --clinic-file data/clinic/clinic_info.json
    python -m src.chunking.chunker --dry-run
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from src.utils.helpers import get_project_root, setup_logging

MAX_SECTION_TOKENS = 500
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

log = setup_logging("chunker")

# ── Tokenizer ────────────────────────────────────────────────────────
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base encoding."""
    return len(_tokenizer.encode(text))


# ── Semantic Splitting ───────────────────────────────────────────────
def semantic_split(content: str) -> list[str]:
    """Split content by markdown section headers.

    Looks for double-newline followed by a capitalized heading line.
    """
    header_pattern = r'\n\n(?=[A-Z][a-zA-Z\s,()-]+(?:\n|$))'
    return re.split(header_pattern, content)


# ── Text splitter fallback for large sections ────────────────────────
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", " "],
)


# ── Medical Chunking ────────────────────────────────────────────────
def chunk_medical_file(file_path: Path) -> list[dict]:
    """Chunk a single cleaned medical article JSON file.

    Args:
        file_path: Path to a JSON file with keys ``title``, ``source``,
                   ``url``, and ``content``.

    Returns:
        List of chunk dicts with ``text`` and ``metadata``.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    title = data.get("title", "")
    source = data.get("source", "")
    url = data.get("url", "")
    content = data.get("content", "")

    sections = semantic_split(content)
    chunks: list[tuple[str, str]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n")
        first_line = lines[0].strip()

        # Detect if first line is a header (short, no period, no dash prefix)
        if len(first_line) < 40 and not first_line.endswith(".") and not first_line.startswith("-"):
            current_section = first_line
            section_content = "\n".join(lines[1:]).strip()
        else:
            current_section = "General"
            section_content = section

        tokens = count_tokens(section_content)
        if tokens <= MAX_SECTION_TOKENS:
            chunks.append((current_section, section_content))
        else:
            sub_chunks = _text_splitter.split_text(section_content)
            for sub_c in sub_chunks:
                chunks.append((current_section, sub_c))

    enriched_chunks: list[dict] = []
    for section_name, chunk_text in chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        formatted_text = (
            f"Document: {title} | Source: {source} | Section: {section_name}\n\n"
            f"Content:\n{chunk_text}"
        )
        enriched_chunks.append({
            "text": formatted_text,
            "metadata": {
                "doc_title": title,
                "source": source,
                "url": url,
                "section": section_name,
                "type": "medical_kb",
            },
        })

    return enriched_chunks


# ── Clinic Chunking ──────────────────────────────────────────────────
def chunk_clinic_data(clinic_file_path: Path) -> list[dict]:
    """Chunk clinic structured data into RAG-ready chunks.

    Args:
        clinic_file_path: Path to ``clinic_info.json``.

    Returns:
        List of chunk dicts with ``text`` and ``metadata``.
    """
    with open(clinic_file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    chunks: list[dict] = []

    # 1. General Clinic Info & Working Hours
    info = data["clinic_info"]
    hours = data["working_hours"]
    hours_str = "\n".join([
        f"- {day.capitalize()}: {h.get('open')} - {h.get('close')}" if h.get("is_open")
        else f"- {day.capitalize()}: Closed"
        for day, h in hours.items()
    ])

    clinic_text = (
        f"Clinic Entity: General Info & Working Hours\n"
        f"Name: {info['name']} ({info['name_en']})\n"
        f"Address: {info['address']}\n"
        f"Phone: {info['phone']} (Emergency: {info['emergency_phone']})\n"
        f"Email: {info['email']}\n"
        f"Working Hours:\n{hours_str}"
    )
    chunks.append({
        "text": clinic_text,
        "metadata": {"type": "clinic_info", "entity": "general_info"},
    })

    # 2. Doctor Profiles
    for doc in data["doctors"]:
        sched_str = "\n".join([
            f"  - {day.capitalize()}: " + ", ".join(times)
            for day, times in doc["schedule"].items() if times
        ])
        doc_text = (
            f"Clinic Doctor Profile ({doc['name']})\n"
            f"ID: {doc['id']}\n"
            f"Name: {doc['name']} ({doc['name_en']})\n"
            f"Specialization: {doc['specialization']} ({doc['specialization_en']})\n"
            f"Consultation Fee: {doc['consultation_fee']} {doc['currency']}\n"
            f"Schedule:\n{sched_str}\n"
            f"Notes: {doc['notes']}"
        )
        chunks.append({
            "text": doc_text,
            "metadata": {
                "type": "clinic_doctor",
                "entity": doc["id"],
                "doctor_name": doc["name_en"],
            },
        })

    # 3. Clinic Services
    for svc in data["services"]:
        duration = f" ({svc['duration_minutes']} min)" if "duration_minutes" in svc else ""
        notes = f" | Notes: {svc['notes']}" if "notes" in svc else ""
        svc_text = (
            f"Clinic Service Details\n"
            f"Service Name: {svc['name']}\n"
            f"Price: {svc['price']} {svc['currency']}{duration}{notes}"
        )
        chunks.append({
            "text": svc_text,
            "metadata": {"type": "clinic_service", "entity": svc["name"]},
        })

    # 4. Clinic Appointment Policies
    policy = data["appointments_policy"]
    policy_text = (
        f"Clinic Appointment Booking & Cancellation Policy\n"
        f"Booking Methods: {', '.join(policy['booking_methods'])}\n"
        f"Cancellation Policy: {policy['cancellation_policy']}\n"
        f"Late Policy: {policy['late_policy']}\n"
        f"Walk-in Policy: {policy['walk_in_policy']}\n"
        f"Average Waiting Time: {policy['average_waiting_time']}"
    )
    chunks.append({
        "text": policy_text,
        "metadata": {"type": "clinic_policy", "entity": "policy"},
    })

    return chunks


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Split cleaned data into chunks for RAG indexing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cleaned-dir",
        default="data/cleaned",
        help="Directory containing cleaned medical JSON files.",
    )
    p.add_argument(
        "--clinic-file",
        default="data/clinic/clinic_info.json",
        help="Path to clinic_info.json.",
    )
    p.add_argument(
        "--output-file",
        default="data/chunks/chunks.json",
        help="Output path for the combined chunks JSON.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report chunk counts without writing to disk.",
    )
    return p.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — chunk medical and clinic data."""
    args = parse_args(argv)
    root = get_project_root()

    cleaned_dir = Path(args.cleaned_dir)
    if not cleaned_dir.is_absolute():
        cleaned_dir = root / cleaned_dir

    clinic_file = Path(args.clinic_file)
    if not clinic_file.is_absolute():
        clinic_file = root / clinic_file

    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = root / output_file

    all_chunks: list[dict] = []

    # 1. Process Medical KB Files
    medical_count = 0
    log.info("Processing Medical Articles from %s ...", cleaned_dir)
    if cleaned_dir.is_dir():
        for json_file in sorted(cleaned_dir.rglob("*.json")):
            file_chunks = chunk_medical_file(json_file)
            all_chunks.extend(file_chunks)
            medical_count += len(file_chunks)
    else:
        log.warning("Cleaned directory not found: %s", cleaned_dir)

    log.info("Generated %d chunks from medical knowledge base.", medical_count)

    # 2. Process Clinic Info File
    log.info("Processing Clinic Data from %s ...", clinic_file)
    if clinic_file.is_file():
        clinic_chunks = chunk_clinic_data(clinic_file)
        all_chunks.extend(clinic_chunks)
        log.info("Generated %d chunks from clinic database.", len(clinic_chunks))
    else:
        log.warning("Clinic info file not found: %s", clinic_file)

    # 3. Assign unique IDs to all chunks
    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{idx}"

    log.info("Total chunks: %d", len(all_chunks))

    if args.dry_run:
        log.info("Dry run — skipping write.")
        return

    # 4. Save to chunks.json
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving %d chunks to %s ...", len(all_chunks), output_file)
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=4)

    log.info("Chunking pipeline completed successfully!")


if __name__ == "__main__":
    main()
