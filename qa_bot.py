#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded Q&A Bot with citations — RAG-ядро.

Пайплайн:
  документи → chunking (по заголовках) → TF-IDF embeddings → векторне сховище
  → запит → embedding → retrieval → grounding → відповідь + citations

Працює повністю офлайн (лише numpy):
  • embeddings   — локальний TF-IDF (клас TfidfEmbedder)
  • vector store — InMemoryVectorStore (косинусний пошук)
  • генерація    — extractive (метод .ask) АБО через Ollama (див. main.py / ollama_llm.py)

Цей файл можна запустити окремо для демонстрації extractive-режиму:
    python3 qa_bot.py
    python3 qa_bot.py "ваше питання"
"""

from __future__ import annotations
import os
import re
import sys
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "docs")

# Поріг grounding: якщо найкращий збіг нижчий — бот каже "не знаю".
SIM_THRESHOLD = 0.22
NOT_FOUND = "Не знайшов інформації у наданій документації."


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
# 3) Embeddings — локальний TF-IDF (без зовнішніх сервісів)
# ==================================================================
# ==================================================================
# 3) Embeddings — локальний TF-IDF (без зовнішніх сервісів)
# ==================================================================

# Список базових англійських стоп-слів, які заважають пошуку
STOP_WORDS = {
    "how", "to", "the", "a", "an", "and", "or", "but", "in", "on", "at", 
    "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "from", "up", 
    "down", "out", "off", "over", "under", "again", "further", 
    "then", "once", "here", "there", "when", "where", "why", "all", 
    "any", "both", "each", "few", "more", "most", "other", "some", "such", 
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", 
    "s", "t", "can", "will", "just", "should", "now", "i", "you", "my"
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", text.lower())
    # Відфільтровуємо службові слова
    return [t for t in tokens if t not in STOP_WORDS]


class TfidfEmbedder:
    """Локальний embedder. Реалізує .fit() і .encode() (як у sklearn, але без залежностей)."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None

    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        doc_freq: Dict[str, int] = {}
        n_docs = len(texts)

        for text in texts:
            for tok in set(tokenize(text)):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq.keys()))}
        idf = np.zeros(len(self.vocab), dtype=np.float64)
        for tok, i in self.vocab.items():
            idf[i] = math.log((1 + n_docs) / (1 + doc_freq[tok])) + 1.0
        self.idf = idf
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder не навчений. Спочатку виклич .fit().")

        vecs = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            tf: Dict[int, int] = {}
            for tok in tokens:
                idx = self.vocab.get(tok)
                if idx is not None:
                    tf[idx] = tf.get(idx, 0) + 1
            for idx, count in tf.items():
                vecs[row, idx] = (count / len(tokens)) * self.idf[idx]

        # L2-нормалізація -> dot product стає косинусною подібністю
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# ==================================================================
# 4) Векторне сховище (in-memory, косинусний пошук)
# ==================================================================
@dataclass
class Hit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        self.embeddings = embeddings
        self.chunks = chunks

    def query(self, qvec: np.ndarray, k: int = 3) -> List[Hit]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        sims = self.embeddings @ qvec  # вектори вже нормалізовані -> це косинус
        order = np.argsort(-sims)[:k]
        return [Hit(chunk=self.chunks[i], score=float(sims[i])) for i in order]


# ==================================================================
# 5) Grounded генерація відповіді + citations (extractive fallback)
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


def generate_answer(query: str, hits: List[Hit], embedder: TfidfEmbedder) -> Answer:
    """Extractive grounded answer: серед речень top-hits обирає найрелевантніше до query."""
    if not hits or hits[0].score < SIM_THRESHOLD:
        return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)

    candidates: List[Tuple[str, str]] = []  # (речення, джерело)
    for hit in hits:
        for sent in split_sentences(hit.chunk.text):
            candidates.append((sent, hit.chunk.source))

    if not candidates:
        return Answer(text=NOT_FOUND, citations=[], grounded=False, hits=hits)

    sent_texts = [c[0] for c in candidates]
    sent_embedder = TfidfEmbedder().fit(sent_texts + [query])
    sent_vecs = sent_embedder.encode(sent_texts)
    q_vec = sent_embedder.encode([query])[0]

    sims = sent_vecs @ q_vec
    best_idx = int(np.argmax(sims))
    best_sentence, _ = candidates[best_idx]

    sources = []
    for hit in hits:
        if hit.chunk.source not in sources:
            sources.append(hit.chunk.source)

    return Answer(text=best_sentence, citations=sources, grounded=True, hits=hits)


# ==================================================================
# 6) Сам бот
# ==================================================================
class GroundedQABot:
    def __init__(self, embedder: Optional[TfidfEmbedder] = None,
                 store: Optional[InMemoryVectorStore] = None,
                 threshold: float = SIM_THRESHOLD):
        self.embedder = embedder or TfidfEmbedder()
        self.store = store or InMemoryVectorStore()
        self.threshold = threshold
        self._indexed = False

    def index(self, docs_dir: str = DOCS_DIR) -> "GroundedQABot":
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)

        if not chunks:
            self.store.add(np.zeros((0, 0)), [])
            self._indexed = True
            return self

        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        embeddings = self.embedder.encode(texts)
        self.store.add(embeddings, chunks)
        self._indexed = True
        return self

    def retrieve(self, question: str, k: int = 3) -> List[Hit]:
        if not self._indexed:
            raise RuntimeError("Бот не проіндексований. Спочатку виклич .index().")
        if not self.store.chunks:
            return []
        qvec = self.embedder.encode([question])[0]
        return self.store.query(qvec, k=k)

    def ask(self, question: str, k: int = 3) -> Answer:
        """Offline extractive-режим (без LLM)."""
        hits = self.retrieve(question, k=k)
        return generate_answer(question, hits, self.embedder)


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
    # За замовчуванням використовуємо стандартну папку docs
    docs_dir = DOCS_DIR
    questions = DEMO_QUESTIONS

    # Якщо передано аргументи командного рядка
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # Якщо перший аргумент — це шлях до існуючої папки
        if os.path.isdir(first_arg):
            docs_dir = first_arg
            # Якщо після папки передані ще аргументи, вважаємо їх окремими питаннями
            if len(sys.argv) > 2:
                questions = sys.argv[2:]
        else:
            # Якщо перший аргумент не є папкою, то вважаємо всі аргументи питаннями
            questions = sys.argv[1:]

    # Індексуємо саме ту папку, яку визначили
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
