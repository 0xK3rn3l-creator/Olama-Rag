#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded Q&A Bot entry point.

Modes:
  1. Offline RAG            — extractive, no LLM
  2. Offline RAG + Ollama   — local retrieval (TF-IDF), generation with Ollama
  3. Normal Ollama          — regular chat without RAG

Ollama is used through the HTTP API with streaming.

Run:
    python3 main.py
"""

from __future__ import annotations
import sys

from qa_bot import GroundedQABot, DOCS_DIR, NOT_FOUND
import ollama_llm as ollama


def _print_header(title: str) -> None:
    print("=" * 40)
    print(title)
    print("=" * 40)


def _choose_mode() -> str:
    _print_header("Grounded QA Bot")
    print()
    print("1. Offline RAG (extractive, без LLM)")
    print("2. Offline RAG + Ollama (grounded generation)")
    print("3. Normal Ollama (без RAG)")
    print()
    while True:
        choice = input("Choose [1/2/3]: ").strip().lower()
        if choice in ("1", "offline"):
            return "offline"
        if choice in ("2", "rag+ollama", "rag"):
            return "rag_ollama"
        if choice in ("3", "ollama", "normal"):
            return "ollama"
        print("Не зрозумів вибір, спробуй ще раз.")


def _choose_docs_path() -> str:
    import os

    path = input(f"Enter path to documents [{DOCS_DIR}]: ").strip()
    if path:
        # Remove quotes, brackets and extra spaces
        path = path.strip('\'"[] \t\n\r\xa0')

        # Expand "~" to the home directory
        path = os.path.expanduser(path)

        # Convert to an absolute path
        path = os.path.abspath(path)

    return path or DOCS_DIR


def _choose_model() -> str:
    default = ollama.DEFAULT_MODEL
    if not ollama.is_ollama_available():
        print("⚠ Команду 'ollama' не знайдено в PATH — переконайся, що вона встановлена.")
        return default
    models = ollama.list_models()
    if models:
        print("Доступні локальні моделі:", ", ".join(models))
    model = input(f"Model [{default}]: ").strip()
    return model or default


def run_offline(docs_path: str) -> None:
    print("\nLoading documents...")
    bot = GroundedQABot().index(docs_path)
    n_docs = len(set(c.source for c in bot.store.chunks))
    print(f"Indexed {len(bot.store.chunks)} chunks from {n_docs} documents.\n")

    if len(bot.store.chunks) == 0:
        print("⚠ Не знайдено жодного .md/.txt файлу за цим шляхом.\n")

    _print_header("Offline RAG Ready")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You > ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        ans = bot.ask(q)
        print("\nAssistant >", ans.text)
        if ans.citations:
            print("   ↳ Sources:", ", ".join(ans.citations))
        if ans.hits:
            print(f"   (score={ans.hits[0].score:.2f}, grounded={ans.grounded})")
        print()


def run_rag_ollama(docs_path: str, model: str) -> None:
    print("\nLoading documents...")
    bot = GroundedQABot().index(docs_path)
    n_docs = len(set(c.source for c in bot.store.chunks))
    print(f"Indexed {len(bot.store.chunks)} chunks from {n_docs} documents.\n")

    if not ollama.is_ollama_available():
        print("⚠ 'ollama' не знайдено в PATH. Встанови й повтори запуск.\n")
        return

    _print_header(f"RAG + Ollama ({model}) Ready")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You > ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        hits = bot.retrieve(q, k=3)

        # Get the best score and check the threshold
        best_score = hits[0].score if hits else 0.0
        is_grounded = len(hits) > 0 and best_score >= bot.threshold

        if not is_grounded:
            print("\nAssistant >", NOT_FOUND)
            print(f"   (score={best_score:.2f}, grounded=False)\n")
            continue

        context_chunks = [f"[{h.chunk.source} / {h.chunk.section}]\n{h.chunk.text}" for h in hits]

        print("\nAssistant > ", end="", flush=True)
        try:
            for token in ollama.ask_ollama_grounded(q, context_chunks, model=model):
                print(token, end="", flush=True)
        except ollama.OllamaError as e:
            print(f"\n⚠ Помилка Ollama: {e}\n")
            continue

        sources = []
        for h in hits:
            if h.chunk.source not in sources:
                sources.append(h.chunk.source)

        # Print sources and similarity score
        print(f"\n   ↳ Sources: {', '.join(sources)}")
        print(f"   (score={best_score:.2f}, grounded=True)\n")


def run_normal_ollama(model: str) -> None:
    if not ollama.is_ollama_available():
        print("⚠ 'ollama' не знайдено в PATH. Встанови й повтори запуск.\n")
        return

    _print_header(f"Normal Ollama ({model})")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You > ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        print("\nAssistant > ", end="", flush=True)
        try:
            for token in ollama.ask_ollama_freeform(q, model=model):
                print(token, end="", flush=True)
        except ollama.OllamaError as e:
            print(f"\n⚠ Помилка Ollama: {e}\n")
            continue
        print("\n")


def main() -> None:
    mode = _choose_mode()

    if mode == "offline":
        docs_path = _choose_docs_path()
        run_offline(docs_path)
    elif mode == "rag_ollama":
        docs_path = _choose_docs_path()
        model = _choose_model()
        run_rag_ollama(docs_path, model)
    elif mode == "ollama":
        model = _choose_model()
        run_normal_ollama(model)

    print("Bye!")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        sys.exit(0)