"""Central prompt templates loaded from versioned text files."""
from pathlib import Path

_ROOT = Path(__file__).parent

def load(name: str) -> str:
    return (_ROOT / f"{name}.txt").read_text(encoding="utf-8")
