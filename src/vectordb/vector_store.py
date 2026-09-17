"""
Module: vector_store.py
Purpose: Encode medical & clinic chunks with BGE-M3 and index them
         into local Qdrant or a configured Qdrant server.

Usage:
    # Index locally with the local FlagEmbedding backend
    python -m src.vectordb.vector_store --local
    python -m src.vectordb.vector_store --mode clean
    python -m src.vectordb.vector_store --chunks-file data/chunks/chunks_v2.json --batch-size 64
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.config import get_settings
from src.embeddings.factory import get_embedder
from src.embeddings.protocol import Embedder
from src.utils.helpers import (
    deterministic_uuid,
    get_git_commit,
    setup_logging,
)

log = setup_logging("vectordb.vector_store")

# ── Defaults ─────────────────────────────────────────────────────────
_DEFAULT_CHUNKS_FILE = "data/chunks/chunks.json"
_DENSE_DIM = 1024
_DEFAULT_BATCH_SIZE = 32
_SCRIPT_VERSION = "2.1.0"


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Index medical/clinic chunks into local Qdrant or a configured Qdrant server. "
            "The --local flag forces on-disk Qdrant and local FlagEmbedding."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--chunks-file",
        default=_DEFAULT_CHUNKS_FILE,
        help="Path to chunks.json (relative to project root or absolute).",
    )
    p.add_argument(
        "--mode",
        choices=["clean", "upsert"],
        default="upsert",
        help=(
            "'clean' = drop existing collections and re-create from scratch. "
            "'upsert' = insert/update into existing collections (default)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Encoding & upsert batch size.",
    )
    p.add_argument(
        "--bge-model",
        default=None,
        help="Override the centralized BGE model setting for this indexing run.",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Override the centralized BGE revision for this indexing run.",
    )
    p.add_argument(
        "--collection-prefix",
        default="",
        help="Optional prefix for collection names (e.g. 'staging_').",
    )
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 for the local embedding model (defaults to BGE_USE_FP16)."
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Force local FlagEmbedding and the on-disk Qdrant database.",
    )
    return p.parse_args(argv)



# ── Chunk Loading ────────────────────────────────────────────────────
def load_chunks(path: Path) -> tuple[list[dict], list[dict]]:
    """Load chunks.json and split into medical vs. clinic chunks.

    Args:
        path: Path to the chunks JSON file.

    Returns:
        Tuple of (medical_chunks, clinic_chunks).

    Raises:
        FileNotFoundError: If the chunks file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        all_chunks: list[dict] = json.load(fh)

    medical = [c for c in all_chunks if c.get("metadata", {}).get("type") == "medical_kb"]
    clinic = [c for c in all_chunks if c.get("metadata", {}).get("type") != "medical_kb"]

    log.info("Loaded %d medical chunks | %d clinic chunks", len(medical), len(clinic))
    return medical, clinic


# ── Collection Management ────────────────────────────────────────────
def ensure_collections(
    client,  # QdrantClient
    collection_names: list[str],
    *,
    mode: str,
) -> None:
    """Create (or recreate) Qdrant collections with dense + optional sparse config.

    Dense (1024) is the primary retrieval vector for the local runtime.
    Sparse remains configured so local FlagEmbedding can write lexical weights.
    """
    from qdrant_client.models import (
        Distance,
        OptimizersConfigDiff,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    existing = {c.name for c in client.get_collections().collections}

    for name in collection_names:
        if mode == "clean" and name in existing:
            log.warning("Mode=clean — deleting collection '%s'", name)
            client.delete_collection(name)
            existing.discard(name)

        if name in existing:
            log.info("Collection '%s' already exists — skipping creation.", name)
            continue

        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        log.info("Created collection: '%s'", name)


# ── Source Merging ───────────────────────────────────────────────────
def _merge_sources(existing_sources: list[str], new_source: str | None) -> list[str]:
    """Return a deduplicated sources list."""
    merged = list(existing_sources)
    if new_source and new_source not in merged:
        merged.append(new_source)
    return merged


def _fetch_existing_payloads(
    client,
    collection_name: str,
    point_ids: list[str],
) -> dict[str, dict]:
    """Retrieve payloads for a batch of point IDs that already exist."""
    if not point_ids:
        return {}
    results = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=True,
        with_vectors=False,
    )
    return {str(r.id): r.payload for r in results if r.payload is not None}


def _dense_to_list(dense) -> list[float]:
    """Convert numpy / list dense vector to a plain float list."""
    if hasattr(dense, "tolist"):
        return dense.tolist()
    return [float(x) for x in dense]


# ── Encode & Index ───────────────────────────────────────────────────
def encode_and_index(
    embedder: Embedder,
    client,  # QdrantClient
    chunks: list[dict],
    collection_name: str,
    *,
    batch_size: int,
    git_commit: str,
) -> int:
    """Encode chunks and upsert into Qdrant in batches (dense-primary).

    Sparse vectors are written only when ``embedder.encode`` returns
    ``lexical_weights`` (local FlagEmbedding). HF API path leaves sparse empty.
    """
    from qdrant_client.models import PointStruct, SparseVector

    if not chunks:
        log.info("No chunks for '%s' — skipping.", collection_name)
        return 0

    texts = [c["text"] for c in chunks]
    total = len(texts)
    indexed_at = datetime.now(timezone.utc).isoformat()

    log.info(
        "Encoding %d chunks for '%s' with %s@%s ...",
        total,
        collection_name,
        embedder.model_id,
        embedder.revision or "default",
    )

    points_upserted = 0
    for i in tqdm(range(0, total, batch_size), desc=f"Indexing {collection_name}"):
        batch_texts = texts[i : i + batch_size]
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = [deterministic_uuid(c["text"]) for c in batch_chunks]

        existing_payloads = _fetch_existing_payloads(client, collection_name, batch_ids)
        if existing_payloads:
            log.debug(
                "Batch %d: %d/%d points already exist — will merge sources.",
                i // batch_size,
                len(existing_payloads),
                len(batch_ids),
            )

        output = embedder.encode(batch_texts, batch_size=batch_size)
        dense_vecs = output["dense_vecs"]
        sparse_vecs = output["lexical_weights"]

        points: list[PointStruct] = []
        for j, (chunk, dense, point_id) in enumerate(
            zip(batch_chunks, dense_vecs, batch_ids)
        ):
            if sparse_vecs is None:
                sparse_indices: list[int] = []
                sparse_values: list[float] = []
            else:
                sparse_weights = sparse_vecs[j]
                sparse_indices = [int(k) for k in sparse_weights.keys()]
                sparse_values = [float(v) for v in sparse_weights.values()]

            incoming_source: str | None = chunk.get("metadata", {}).get("source")

            if point_id in existing_payloads:
                existing_sources: list[str] = existing_payloads[point_id].get("sources", [])
                if not existing_sources:
                    old_source = existing_payloads[point_id].get("source")
                    existing_sources = [old_source] if old_source else []
                merged_sources = _merge_sources(existing_sources, incoming_source)
            else:
                merged_sources = [incoming_source] if incoming_source else []

            meta = {k: v for k, v in chunk.get("metadata", {}).items() if k != "source"}

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": _dense_to_list(dense),
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload={
                        "text": chunk["text"],
                        "chunk_id": chunk.get("id", point_id),
                        **meta,
                        "sources": merged_sources,
                        "indexed_at": indexed_at,
                        "git_commit": git_commit,
                        "script_version": _SCRIPT_VERSION,
                        "embed_model_id": embedder.model_id,
                        "embed_revision": embedder.revision or "",
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        points_upserted += len(points)

    log.info("Upserted %d points into '%s'", points_upserted, collection_name)
    return points_upserted


# ── Verification ─────────────────────────────────────────────────────
def verify_collections(client, collection_names: list[str]) -> None:
    """Log point counts and status for the given collections."""
    log.info("─── Collection Stats ───────────────────────")
    for name in collection_names:
        info = client.get_collection(name)
        log.info("  %s: %d points | status: %s", name, info.points_count, info.status)
    log.info("────────────────────────────────────────────")


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — parse args, load env, encode, and index."""
    args = parse_args(argv)
    settings = get_settings()
    if args.local:
        settings = settings.for_local()

    root = settings.project_root
    git_commit = get_git_commit()
    log.info("Project root: %s", root)
    log.info("Git commit:   %s", git_commit)

    backend = settings.embedding.backend.strip().lower()

    if backend not in {"modal", "modal_api", "remote", "local", "flag", "flagembedding"}:
        log.error(
            "Unknown or unsupported EMBEDDER_BACKEND=%r. "
            "Supported backends: 'local' (default FlagEmbedding) or 'modal'.",
            backend,
        )
        sys.exit(1)

    chunks_path = Path(args.chunks_file)
    if not chunks_path.is_absolute():
        chunks_path = root / chunks_path

    prefix = args.collection_prefix
    medical_coll = f"{prefix}{settings.vectordb.medical_collection}"
    clinic_coll = f"{prefix}{settings.vectordb.clinic_collection}"
    collection_names = [medical_coll, clinic_coll]

    qdrant_url = settings.vectordb.qdrant_url.strip().rstrip("/")
    qdrant_path = settings.resolve_path(settings.vectordb.qdrant_path)
    qdrant_api_key = settings.secrets.qdrant_api_key or None

    t_start = time.time()

    medical_chunks, clinic_chunks = load_chunks(chunks_path)

    from qdrant_client import QdrantClient

    if qdrant_url:
        qdrant_dest = qdrant_url
        log.info("Connecting to Qdrant server at %s ...", qdrant_url)
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_path_obj = Path(qdrant_path).expanduser()
        if not qdrant_path_obj.is_absolute():
            qdrant_path_obj = root / qdrant_path_obj
        qdrant_path_obj.mkdir(parents=True, exist_ok=True)
        qdrant_dest = str(qdrant_path_obj)
        log.info("Opening local on-disk Qdrant at %s ...", qdrant_path_obj)
        client = QdrantClient(path=str(qdrant_path_obj))

    log.info("Connected to Qdrant at %s.", qdrant_dest)


    ensure_collections(client, collection_names, mode=args.mode)

    model_id = args.bge_model or settings.embedding.model_id
    revision = (
        args.revision
        if args.revision is not None
        else settings.embedding.revision
    )
    embed_backend = backend
    use_fp16 = args.fp16 if args.fp16 is not None else settings.embedding.use_fp16

    log.info(
        "Loading embedder backend=%s model=%s revision=%s fp16=%s ...",
        embed_backend,
        model_id,
        revision or "default",
        use_fp16,
    )
    embedder = get_embedder(
        backend=embed_backend,
        model_id=model_id,
        revision=revision or None,
        use_fp16=use_fp16,
    )
    log.info("Embedder ready.")


    encode_and_index(
        embedder=embedder,
        client=client,
        chunks=medical_chunks,
        collection_name=medical_coll,
        batch_size=args.batch_size,
        git_commit=git_commit,
    )
    encode_and_index(
        embedder=embedder,
        client=client,
        chunks=clinic_chunks,
        collection_name=clinic_coll,
        batch_size=args.batch_size,
        git_commit=git_commit,
    )

    verify_collections(client, collection_names)

    elapsed = time.time() - t_start
    log.info("Done. Total time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
