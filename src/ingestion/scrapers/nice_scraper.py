"""
Module: nice_scraper.py
Purpose: Scrape NICE clinical guideline recommendation pages for medical RAG data.

Usage:
    python -m src.ingestion.scrapers.nice_scraper
    python -m src.ingestion.scrapers.nice_scraper --output-dir data/raw/nice --delay 3
"""

import argparse
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.helpers import get_project_root, setup_logging

log = setup_logging("nice_scraper")

# ── Default Guidelines ─────────────────────────────────────────────────────
# Each entry maps: guideline_id -> (topic_slug, full_name)
_DEFAULT_GUIDELINES: dict[str, tuple[str, str]] = {
    "ng136": ("hypertension",         "Hypertension in adults: diagnosis and management"),
    "ng28":  ("type-2-diabetes",       "Type 2 diabetes in adults: management"),
    "ng19":  ("diabetic-foot",         "Diabetic foot problems: prevention and management"),
    "ng80":  ("asthma",               "Asthma: diagnosis, monitoring and chronic asthma management"),
    "ng115": ("copd",                 "Chronic obstructive pulmonary disease in over 16s: diagnosis and management"),
    "ng222": ("depression",           "Depression in adults: treatment and management"),
    "ng203": ("chronic-kidney-disease","Chronic kidney disease: assessment and management"),
    "ng226": ("osteoarthritis",       "Osteoarthritis in over 16s: diagnosis and management"),
    "ng196": ("atrial-fibrillation",  "Atrial fibrillation: diagnosis and management"),
    "cg150": ("headaches",            "Headaches in over 12s: diagnosis and management"),
    "ng219": ("gout",                 "Gout: diagnosis and management"),
    "ng91":  ("otitis-media",         "Otitis media (acute): antimicrobial prescribing"),
    "ng109": ("urinary-tract-infection","Urinary tract infection (lower) - acute: antimicrobial prescribing"),
    "ng84":  ("sore-throat",          "Sore throat (acute): antimicrobial prescribing"),
    "ng120": ("cough",                "Cough (acute): antimicrobial prescribing"),
    "ng79":  ("sinusitis",            "Sinusitis (acute): antimicrobial prescribing"),
    "ng59":  ("low-back-pain",        "Low back pain and sciatica in over 16s: assessment and management"),
    "cg184": ("gord-dyspepsia",       "Gastro-oesophageal reflux disease and dyspepsia in adults: investigation and management"),
}


# ── HTTP Session ─────────────────────────────────────────────────────
def _create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "close",
    })
    return session


# ── Scraping ─────────────────────────────────────────────────────────
def scrape_nice_recommendations(
    session: requests.Session, guideline_id: str
) -> dict | None:
    """Scrape the recommendations chapter of a NICE guideline.

    Args:
        session: Configured requests session.
        guideline_id: NICE guideline ID (e.g. ``'ng136'``).

    Returns:
        Dict with ``title``, ``url``, ``source``, ``content``, or ``None`` on failure.
    """
    url = f"https://www.nice.org.uk/guidance/{guideline_id}/chapter/Recommendations"

    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
    except Exception as e:
        log.error("Failed to fetch %s: %s", url, e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.find(class_="chapter") or soup.find("main")
    if not content_div:
        log.warning("Could not find chapter or main content in %s", url)
        return None

    # Remove unwanted elements
    for element in content_div.find_all(["script", "style", "nav", "footer", "aside"]):
        element.decompose()

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else f"NICE Guideline {guideline_id.upper()}"

    # If the title is just "Recommendations", prefix with guideline ID
    if title.lower() == "recommendations":
        title = f"{guideline_id.upper()} Recommendations"

    text = content_div.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return {
        "title": title,
        "url": url,
        "source": "NICE Guidelines",
        "content": clean_text,
    }


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Scrape NICE clinical guideline recommendations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/nice",
        help="Output directory for scraped JSON files.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between requests.",
    )
    return p.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — scrape NICE guideline recommendations."""
    args = parse_args(argv)
    root = get_project_root()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _create_session()
    guidelines = _DEFAULT_GUIDELINES

    for i, (gid, (slug, name)) in enumerate(guidelines.items()):
        log.info("Scraping (%d/%d): %s (%s)", i + 1, len(guidelines), gid, name)

        data = scrape_nice_recommendations(session, gid)
        if data:
            filepath = output_dir / f"{slug}.json"
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            log.info("Saved: %s", filepath.name)
        else:
            log.warning("Failed to scrape guideline: %s", gid)

        time.sleep(args.delay)

    log.info("Done! All NICE guidelines saved to %s", output_dir)


if __name__ == "__main__":
    main()
