"""
Vector store for caching 3D generation results by image embedding similarity.
Uses ChromaDB with cosine distance for fast nearest-neighbor lookup.
"""
import hashlib
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "generated/vector_store",
        collection_name: str = "generation_cache",
        similarity_threshold: float = 0.95,
    ):
        import chromadb

        self.similarity_threshold = similarity_threshold
        self._lock = threading.Lock()

        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready: %s (%d entries, threshold=%.2f)",
            persist_dir, self._collection.count(), similarity_threshold,
        )

    def search(self, embedding: List[float], params_hash: str) -> Optional[Dict]:
        """Search for a cached result with similar embedding and matching params."""
        if self._collection.count() == 0:
            return None

        max_distance = 1.0 - self.similarity_threshold

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"params_hash": params_hash},
        )

        if (
            results["ids"]
            and results["ids"][0]
            and results["distances"][0][0] <= max_distance
        ):
            meta = results["metadatas"][0][0]
            return {
                "id": results["ids"][0][0],
                "distance": results["distances"][0][0],
                "result_json": meta.get("result_json", "{}"),
                "source": meta.get("source", "unknown"),
                "created_at": meta.get("created_at", ""),
            }
        return None

    def store(
        self,
        embedding: List[float],
        params_hash: str,
        result: Dict,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store a generation result with its embedding."""
        entry_id = str(uuid.uuid4())
        meta = {
            "params_hash": params_hash,
            "result_json": json.dumps(result),
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        with self._lock:
            self._collection.add(
                ids=[entry_id],
                embeddings=[embedding],
                metadatas=[meta],
            )
        logger.info("Stored cache entry %s (params=%s)", entry_id[:8], params_hash[:8])
        return entry_id

    def list_all(self) -> List[Dict]:
        """Return all cache entries."""
        data = self._collection.get(include=["metadatas"])
        entries = []
        for i, entry_id in enumerate(data["ids"]):
            meta = data["metadatas"][i] if data["metadatas"] else {}
            entries.append({"id": entry_id, **meta})
        return entries

    def delete(self, entry_id: str) -> bool:
        """Delete a cache entry by ID."""
        try:
            with self._lock:
                self._collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    @staticmethod
    def compute_params_hash(**kwargs) -> str:
        """Deterministic hash of generation parameters."""
        canonical = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
