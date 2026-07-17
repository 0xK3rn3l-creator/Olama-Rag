# 🤖 Grounded Q&A Bot (RAG with Ollama)

**Grounded Q&A Bot** is a simple RAG (Retrieval-Augmented Generation) application. It can search your local text files (`.txt`, `.md`) and answer questions using the information inside them.

Everything runs on your own computer. Your documents stay local, and no data is sent to the cloud.

---

## 🚀 Getting Started

Before running the project, install the required software and prepare your environment.

### 1. Install Ollama

Ollama is used for the modes that generate answers with an LLM.

1. Download and install Ollama from **https://ollama.com**.
2. Start the Ollama application.
3. Download a model, for example:

```bash
ollama pull llama3:8b
```

You can also use a smaller model such as `gemma2:2b`.

### 2. Create a Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# Windows:
# venv\Scripts\activate
```

Install NumPy:

```bash
pip install numpy
```

### 3. Add Your Documents

Create a `docs` folder and put your `.md` or `.txt` files inside it.

Example:

```bash
mkdir docs
echo "## Password Reset\nTo reset your password, click on 'Forgot Password' on the login page." > docs/auth.md
```

---

## 🛠️ Available Modes

Run the program with:

```bash
python3 main.py
```

Choose one of the three modes.

### 1️⃣ Offline RAG

**How it works**

* Splits documents into chunks.
* Creates TF-IDF embeddings.
* Finds the best matching sentence.
* Returns the answer directly from the documents.

**Pros**

* Very fast.
* Works without Ollama.
* No GPU needed.

---

### 2️⃣ Offline RAG + Ollama

**How it works**

* Searches your documents for the best matching text.
* Sends the found context to Ollama.
* Generates a grounded answer using only the retrieved information.
* If nothing relevant is found, the bot says that it does not know.

**Pros**

* More natural answers.
* Uses your own documents as the knowledge source.
* Supports streaming output.

---

### 3️⃣ Normal Ollama

**How it works**

* Sends your question directly to Ollama.
* Does not use the document database.

**Pros**

* Simple chat with a local LLM.
* Good for general questions.

---

## 📂 Project Files

The project has four main files.

* `main.py` – starts the program, lets you choose the mode, loads documents, and prints streaming output.
* `qa_bot.py` – loads documents, splits them into chunks, creates TF-IDF embeddings, and searches for the best match.
* `ollama_llm.py` – connects to Ollama through the HTTP API and supports streaming with CLI fallback.
* `README.md` – project documentation.

---

## 💻 Example

```text
========================================
Grounded QA Bot
========================================

1. Offline RAG
2. Offline RAG + Ollama
3. Normal Ollama

Choose [1/2/3]: 2
Enter path to documents [/user/projects/docs]:
Available models: llama3:8b, gemma2:2b
Model [llama3:8b]: llama3:8b

Loading documents...
Indexed 12 chunks from 3 documents.

========================================
RAG + Ollama (llama3:8b) Ready
========================================
Type 'exit' to quit.

You > How do I reset my password?

Assistant > To reset your password, click on the "Forgot Password" link on the login page and enter your registered email address.
   ↳ Sources: auth.md
   (score=0.45, grounded=True)
```

---

## ⚙️ Settings

You can change the search threshold in `qa_bot.py`.

* `SIM_THRESHOLD = 0.22`

If the similarity score is lower than this value, the bot will return that it could not find the answer in your documents instead of making one up.
