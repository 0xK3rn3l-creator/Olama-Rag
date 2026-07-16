#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded Q&A Bot with citations — RAG-ядро на базі ChromaDB.

Пайплайн:
  документи → chunking (по заголовках) → ChromaDB (локальні семантичні embeddings)
  → запит → embedding → retrieval → grounding → відповідь + citations

Працює повністю офлайн (за допомогою ChromaDB ONNXMiniLM):
  • embeddings   — вбудовані семантичні ONNX-ембедінги
  • vector store — ChromaDB (збереження бази у директорію chroma_db)
  • генерація    — extractive (метод .ask) АБО через Ollama (див. main.py / ollama_llm.py)
"""

from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
# Імпортуємо ChromaDB та її швидкі локальні ONNX-ембедінги
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "docs")
CHROMA_DIR = os.path.join(BASE, "chroma_db")

# Поріг grounding: якщо найкращий збіг нижчий — бот каже "не знаю".
# Для семантичних векторів зазвичай використовується діапазон схожості 0.35 - 0.45.
SIM_THRESHOLD = 0.40
NOT_FOUND = "Не знайшел інформації у наданій документації."


# ==================================================================
# 1) Завантаження документів
# ==================================================================
def load_documents(docs_dir: str) -> List[Dict]:
    """Рекурсивно читає всі .md / .txt файли з директорії."""
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
# 2) Chunking — по Markdown-заголовках "## Section"
# ==================================================================
@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_documents(docs: List[Dict]) -> List[Chunk]:
    """Ріже кожен документ по заголовках. Якщо заголовків немає — весь
    документ стає одним чанком."""
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

        # Текст перед першим заголовком
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
# 3) Embeddings & 4) Векторне сховище (Переписано на ChromaDB)
# ==================================================================
@dataclass
class Hit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    """
    Збережено для сумісності інтерфейсів з іншими модулями.
    Тепер працює як інтерфейс до ChromaDB.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []


# ==================================================================
# 5) Grounded генерація відповіді (Оновлено під семантичні ембедінги)
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
    """Extractive grounded answer за допомогою вбудованої моделі ембедінгів Chroma."""
    if not hits or hits[0].score < SIM_THRESHOLD:
        return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)

    candidates: List[Tuple[str, str]] = []  # (речення, джерело)
    for hit in hits:
        for sent in split_sentences(hit.chunk.text):
            if len(sent.split()) > 2:  # Ігноруємо занадто короткі уривки
                candidates.append((sent, hit.chunk.source))

    if not candidates:
        return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)

    sent_texts = [c[0] for c in candidates]
    
    # Використовуємо локальний ONNX MiniLM для точного ранжування речень
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
# 6) Сам бот (ChromaDB інтеграція)
# ==================================================================
class GroundedQABot:
    def __init__(self, embedder=None, store=None, threshold: float = SIM_THRESHOLD):
        self.threshold = threshold
        
        # Ініціалізуємо локальну модель семантичних ембедінгів Chroma (~80MB)
        self.emb_fn = ONNXMiniLM_L6_V2()
        
        # Налаштовуємо базу даних з персистентним збереженням на диску
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Створюємо або отримуємо колекцію з косинусною відстанью (cosine space)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents_collection",
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._indexed = self.collection.count() > 0

    @property
    def store(self):
        """Емулюємо властивості оригінального VectorStore для сумісності з іншими скриптами."""
        class DummyStore:
            def __init__(self, count, chunks):
                self.chunks = chunks
        
        # Створюємо фіктивні Chunk об'єкти для відображення статистики в main()
        dummy_chunks = []
        if self.collection.count() > 0:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            if metas:
                for m in metas:
                    dummy_chunks.append(Chunk(id=m["original_id"], source=m["source"], section=m["section"], text=""))
        return DummyStore(self.collection.count(), dummy_chunks)

    def index(self, docs_dir: str = DOCS_DIR) -> "GroundedQABot":
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)

        if not chunks:
            self._indexed = True
            return self

        # Підготовка пакетів даних для ChromaDB
        ids = [f"chunk_{c.id}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source": c.source, "section": c.section, "original_id": c.id} for c in chunks]

        # Chroma автоматично викличе ONNX модель та збереже вектори у базу
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

        # Пошук найближчих векторів
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
                # Перетворюємо косинусну відстань у схожість (Similarity score)
                score = 1.0 - float(distances[i])
                hits.append(Hit(chunk=chunk, score=score))

        return hits

    def ask(self, question: str, k: int = 3) -> Answer:
        """Offline extractive-режим за допомогою семантичних ембедінгів Chroma."""
        hits = self.retrieve(question, k=k)
        return generate_answer(question, hits, self.emb_fn)


# ==================================================================
# 7) Демонстрація (можна запускати цей файл окремо)
# ==================================================================
DEMO_QUESTIONS = [
    "How do I reset my password?",
    "How can I create an invoice?",
    "How do I delete a user?",
    "How does API authentication work?",
    "Which currencies are supported?",
    "Do you have a mobile app?",   # немає в документації -> grounding спрацює
]


def _demo():
    docs_dir = DOCS_DIR
    questions = DEMO_QUESTIONS

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if os.path.isdir(first_arg):
            docs_dir = first_arg
            if len(sys.argv) > 2:
                questions = sys.argv[2:]
        else:
            questions = sys.argv[1:]

    # Створюємо бота та запускаємо індексацію
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