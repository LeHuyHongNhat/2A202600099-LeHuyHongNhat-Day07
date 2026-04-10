from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    - persist_directory=None  → in-memory list (default, isolated, used by tests)
    - persist_directory="..." → ChromaDB PersistentClient; embeddings survive
                                 across runs — API calls are skipped on reload.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        if persist_directory is not None:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(persist_directory))
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._use_chroma = True
                print(f"[ChromaDB] persist={persist_directory!r}  "
                      f"collection={collection_name!r}  "
                      f"stored={self._collection.count()} chunks")
            except Exception as exc:
                print(f"[ChromaDB] Không khởi động được ({exc}), dùng in-memory.")
                self._use_chroma = False
                self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": {**doc.metadata, "doc_id": doc.id},
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = [
            (record, _dot(query_embedding, record["embedding"]))
            for record in records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**record, "score": score}
            for record, score in scored[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            if self._use_chroma and self._collection is not None:
                # upsert: thêm mới hoặc cập nhật nếu ID đã tồn tại
                self._collection.upsert(
                    ids=[record["id"]],
                    documents=[record["content"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]],
                )
            else:
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
            )
            return [
                {"id": id_, "content": doc, "score": 1 - dist, "metadata": meta}
                for id_, doc, dist, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                )
            ]
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=metadata_filter,
            )
            return [
                {"id": id_, "content": doc, "score": 1 - dist, "metadata": meta}
                for id_, doc, dist, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                )
            ]

        filtered = [
            record for record in self._store
            if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            results = self._collection.get(where={"doc_id": doc_id})
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return bool(ids)

        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
