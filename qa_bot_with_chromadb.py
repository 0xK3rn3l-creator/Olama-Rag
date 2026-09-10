#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded Q&A Bot with citations using ChromaDB.

Pipeline:
  documents → chunking (by headings) → ChromaDB (local semantic embeddings)
  → query → embedding → retrieval → grounded answer + citations

Runs fully offline with ChromaDB ONNXMiniLM:
  • embeddings   — built-in ONNX embeddings
  • vector store — ChromaDB (saved in the chroma_db folder)
  • generation   — extractive (.ask) or Ollama (see main.py / ollama_llm.py)
"""

from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

# ChromaDB and local ONNX embeddings
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "docs")
CHROMA_DIR = os.path.join(BASE, "chroma_db")

SIM_THRESHOLD = 0.40
NOT_FOUND = "Не знайшел інформації у наданій документації."


# ==================================================================
# 1) Load documents
# ==================================================================
def load_documents(docs_dir: str) -> List[Dict]:
    """Read all .md and .txt files from a folder."""
    docs: List[Dict] = []
    if not os.path.isdir(docs_dir):
        return docs

    for root, _, files in os.walk(docs_dir):
        for fname in sorted(files):
            if fname.lower().endswith((".md", ".txt")):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue
                docs.append({
                    "source": os.path.relpath(fpath, docs_dir),
                    "text": text,
                })
    return docs


# ==================================================================
# 2) Split documents into chunks
# ==================================================================
@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_documents(docs: List[Dict]) -> List[Chunk]:
    """Split each document by Markdown headings. If there are no headings,
    the whole document becomes one chunk."""
    chunks: List[Chunk] = []
    cid = 0

    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        headings = list(_HEADING_RE.finditer(text))

        if not headings:
            body = text.strip()
            if body:
                chunks.append(Chunk(id=cid, source=source, section="(document)", text=body))
                cid += 1
            continue

        # Text before the first heading
        if headings[0].start() > 0:
            intro = text[: headings[0].start()].strip()
            if intro:
                chunks.append(Chunk(id=cid, source=source, section="Intro", text=intro))
                cid += 1

        for i, m in enumerate(headings):
            section_title = m.group(2).strip()
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if body:
                chunks.append(Chunk(id=cid, source=source, section=section_title, text=body))
                cid += 1

    return chunks


# ==================================================================
# 3) Embeddings & 4) Vector store
# ==================================================================
@dataclass
class Hit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    """
    Kept for compatibility with the other modules.
    Now it works as a ChromaDB interface.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []


# ==================================================================
# 5) Generate grounded answer
# ==================================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]


@dataclass
class Answer:
    text: str
    citations: List[str] = field(default_factory=list)
    grounded: bool = True
    hits: List[Hit] = field(default_factory=list)

    def __str__(self) -> str:
        out = self.text
        if self.citations:
            out += "\n   ↳ Source: " + "; ".join(self.citations)
        return out


def generate_answer(query: str, hits: List[Hit], emb_fn) -> Answer:
    """Pick the best matching sentence using Chroma embeddings."""
    if not hits or hits[0].score < SIM_THRESHOLD:
        return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)

    candidates: List[Tuple[str, str]] = []  # (sentence, source)
    for hit in hits:
        for sent in split_sentences(hit.chunk.text):
            if len(sent.split()) > 2:  # Skip very short sentences
                candidates.append((sent, hit.chunk.source))

    if not candidates:
        return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)

    sent_texts = [c[0] for c in candidates]

    # Create embeddings for sentence ranking
    candidate_embeddings = np.array(emb_fn(sent_texts))
    query_embedding = np.array(emb_fn([query])[0])

    norm_candidates = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    norm_query = query_embedding / np.linalg.norm(query_embedding)

    sims = norm_candidates @ norm_query
    best_idx = int(np.argmax(sims))
    best_sentence, _ = candidates[best_idx]

    sources = []
    for hit in hits:
        if hit.chunk.source not in sources:
            sources.append(hit.chunk.source)

    return Answer(text=best_sentence, citations=sources, grounded=True, hits=hits)
# ==================================================================
# 6) Main bot
# ==================================================================
class GroundedQABot:
    def __init__(self, embedder=None, store=None, threshold: float = SIM_THRESHOLD):
        self.threshold = threshold

        # Initialize the local embedding model (~80 MB)
        self.emb_fn = ONNXMiniLM_L6_V2()

        # Create a persistent ChromaDB database
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Create or load the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents_collection",
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._indexed = self.collection.count() > 0

    @property
    def store(self):
        """Provide the same interface as the original VectorStore."""

        class DummyStore:
            def __init__(self, count, chunks):
                self.chunks = chunks

        # Create temporary Chunk objects for statistics
        dummy_chunks = []
        if self.collection.count() > 0:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            if metas:
                for m in metas:
                    dummy_chunks.append(
                        Chunk(
                            id=m["original_id"],
                            source=m["source"],
                            section=m["section"],
                            text=""
                        )
                    )
        return DummyStore(self.collection.count(), dummy_chunks)

    def index(self, docs_dir: str = DOCS_DIR) -> "GroundedQABot":
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)

        if not chunks:
            self._indexed = True
            return self

        # Prepare data for ChromaDB
        ids = [f"chunk_{c.id}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "section": c.section,
                "original_id": c.id
            }
            for c in chunks
        ]

        # Chroma creates embeddings automatically
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        self._indexed = True
        return self

    def retrieve(self, question: str, k: int = 3) -> List[Hit]:
        if not self._indexed:
            raise RuntimeError("Бот не проіндексований. Спочатку виклич .index().")
        if self.collection.count() == 0:
            return []

        # Search for the closest vectors
        results = self.collection.query(
            query_texts=[question],
            n_results=k
        )

        hits = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                chunk = Chunk(
                    id=metadatas[i]["original_id"],
                    source=metadatas[i]["source"],
                    section=metadatas[i]["section"],
                    text=documents[i]
                )

                # Convert distance to similarity score
                score = 1.0 - float(distances[i])
                hits.append(Hit(chunk=chunk, score=score))

        return hits

    def ask(self, question: str, k: int = 3) -> Answer:
        """Offline extractive mode using Chroma embeddings."""
        hits = self.retrieve(question, k=k)
        return generate_answer(question, hits, self.emb_fn)


# ==================================================================
# 7) Demo
# ==================================================================
DEMO_QUESTIONS = [
    "How do I reset my password?",
    "How can I create an invoice?",
    "How do I delete a user?",
    "How does API authentication work?",
    "Which currencies are supported?",
    "Do you have a mobile app?",
]


def _demo():
    docs_dir = DOCS_DIR
    questions = DEMO_QUESTIONS

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]

        # If the first argument is a folder
        if os.path.isdir(first_arg):
            docs_dir = first_arg

            # Use the remaining arguments as questions
            if len(sys.argv) > 2:
                questions = sys.argv[2:]
        else:
            questions = sys.argv[1:]

    # Create the bot and index the documents
    bot = GroundedQABot().index(docs_dir)

    print(f"Проіндексовано чанків: {len(bot.store.chunks)} "
          f"з {len(set(c.source for c in bot.store.chunks))} документів у папці: {docs_dir}\n")

    for q in questions:
        ans = bot.ask(q)
        print("Q:", q)
        print("A:", ans.text)
        if ans.citations:
            print("   ↳ Source:", "; ".join(ans.citations))
        if ans.hits:
            print(f"   (score={ans.hits[0].score:.2f}, grounded={ans.grounded})")
        print()


if __name__ == "__main__":
    _demo()