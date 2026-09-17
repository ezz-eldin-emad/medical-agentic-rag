"""
Modal Serverless Deployment for BGE-M3 Embedding Model.

Deploys a FastAPI HTTP POST endpoint on a Modal T4 GPU container.
Exposes dense (1024-dim), sparse (lexical weights), and optional ColBERT embeddings.

Deployment Command:
    modal deploy modal/app.py
"""

import hmac
import os
from typing import Any, List

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

app = modal.App("bge-m3-api")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "numpy",
        "fastapi[standard]",
    )
)


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=32)
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = False


@app.cls(image=image, gpu="T4", secrets=[modal.Secret.from_name("medical-rag-secrets")])
class BGEM3API:
    @modal.enter()
    def load_model(self) -> None:
        """Load BGE-M3 model into GPU memory on container startup."""
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    @modal.fastapi_endpoint(method="POST")
    def embed(
        self,
        req: EmbedRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Generate BGE-M3 embeddings for requested texts."""
        expected = os.environ.get("MODAL_EMBED_TOKEN", "")
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not expected or not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if any(len(text) > 8192 for text in req.texts):
            raise HTTPException(status_code=413, detail="Input text is too long")
        output = self.model.encode(
            req.texts,
            return_dense=req.return_dense,
            return_sparse=req.return_sparse,
            return_colbert_vecs=req.return_colbert,
        )

        result: dict[str, Any] = {}

        if req.return_dense:
            result["dense"] = output["dense_vecs"].tolist()

        if req.return_sparse:
            result["sparse"] = [
                {k: float(v) for k, v in w.items()}
                for w in output["lexical_weights"]
            ]

        if req.return_colbert and "colbert_vecs" in output:
            result["colbert"] = [
                doc_vectors.tolist()
                for doc_vectors in output["colbert_vecs"]
            ]

        return result
