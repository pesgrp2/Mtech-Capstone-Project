# AI Product Review Analyzer (RAG)

A **Streamlit** app that answers natural-language questions over customer reviews using **Retrieval-Augmented Generation (RAG)**. It retrieves relevant review chunks from a **FAISS** vector index, generates an answer with **Ollama** (`llama3`), shows **recommended products**, **ROUGE-style overlap** metrics (optional), and a **product summary** dialog backed by review text and optional Amazon-style metadata.

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **[Ollama](https://ollama.com/)** installed and running locally
- **~2–4 GB disk** for embedding models and the Llama 3 weights (first pull)

---

## 1. Install Ollama and the `llama3` model

### Install Ollama

- **macOS / Windows:** Download and install from [ollama.com/download](https://ollama.com/download).
- **Linux:** follow the instructions on [ollama.com](https://ollama.com/) (install script or package).

### Start the Ollama server

Ollama usually runs as a background service after install. If you need to start it manually:

```bash
ollama serve
```

Leave this terminal open, or run it as a system service (depends on your OS). The app talks to Ollama at the default URL (`http://127.0.0.1:11434`).

### Pull the `llama3` model (required)

In a **separate** terminal:

```bash
ollama pull llama3
```

Verify the model is available:

```bash
ollama list
```

You should see `llama3` in the list. The app uses this model name by default (see `OLLAMA_MODEL` in `review_app/config.py`).

**Optional quick test:**

```bash
ollama run llama3 "Say hello in one sentence."
```

---

## 2. Python environment and dependencies

From the **project root** (`Mtech-Capstone-Project/`):

```bash
python -m venv venv
```

Activate the venv:

- **macOS / Linux:** `source venv/bin/activate`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`

Install packages:

```bash
pip install -r requirement.txt
```

---

## 3. Data and FAISS index

### Review data (parquet)

By default the app loads:

| Role | Default path | Override env var |
|------|----------------|------------------|
| Embedded reviews for RAG / summaries | `embedding_ready_reviews_small.parquet` | `REVIEWS_PARQUET` |

The parquet should include at least `asin`, and text in a column such as `embedding_text` (see `review_app/data/` for fallbacks).

### Product metadata (optional, for images and dialog details)

| Role | Default path | Override env var |
|------|----------------|------------------|
| Amazon-style meta CSV | `src/chinmay/Appliances_meta.csv` | `META_CSV` |

### Vector index directory

| Role | Default | Override env var |
|------|---------|------------------|
| Saved FAISS index | `faiss_index/` | `FAISS_INDEX_DIR` |

### Build the FAISS index (first time only)

If `faiss_index/` is missing, build it **once**:

**macOS / Linux:**

```bash
export REVIEWS_PARQUET="embedding_ready_reviews_small.parquet"   # optional if default is fine
BUILD_FAISS_INDEX=1 streamlit run app.py
```

**Windows (Command Prompt):**

```cmd
set REVIEWS_PARQUET=embedding_ready_reviews_small.parquet
set BUILD_FAISS_INDEX=1
streamlit run app.py
```

When you see the message that the index was built, **stop** the app, **unset** `BUILD_FAISS_INDEX`, and run normally (next section).

---

## 4. Run the Streamlit app

**Requirements before this step:** Ollama is running and `ollama pull llama3` has completed; `faiss_index/` exists; parquet (and optional meta CSV) are in place.

From the project root, with venv activated:

```bash
streamlit run app.py
```

The UI opens in the browser (typically [http://localhost:8501](http://localhost:8501)).

### Optional: theme

Streamlit theme defaults live in `.streamlit/config.toml`.

---

## Project structure (for developers)

High-level rule: **`app.py` is the entry point**; **domain logic lives in `review_app/`**.

```
Mtech-Capstone-Project/
├── app.py                      # Streamlit entry: form, RAG invoke, results, charts, dialog trigger
├── requirement.txt             # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Streamlit theme (light / primary color)
│
├── review_app/                 # Application package (import as review_app.*)
│   ├── __init__.py
│   ├── config.py               # Paths, env defaults, model names, UI/retrieval constants
│   ├── evaluation.py         # ROUGE vs retrieved text (optional metric)
│   ├── rag.py                  # Embeddings, FAISS load, Ollama LLM, RetrievalQA chain, per-product LLM summary
│   ├── build_index.py          # One-shot FAISS build when BUILD_FAISS_INDEX=1
│   ├── ui.py                   # Custom CSS, sidebar copy, product cards, “More” list, @st.dialog summary
│   └── data/                   # Data access layer (cached loads + business helpers)
│       ├── __init__.py         # Re-exports main data API
│       ├── images.py           # Parse/sanitize image URLs from meta strings
│       ├── loading.py          # @st.cache_data: parquet + meta CSV
│       ├── catalog.py          # ASIN → title/image from parquet + meta
│       ├── meta.py             # Rich meta row for dialog (price, ratings, gallery URLs)
│       ├── reviews.py          # Collect review text per ASIN; extractive card snippet
│       └── ranking.py          # Rank ASINs from similarity_search_with_score
│
├── faiss_index/                # Generated FAISS files (not always in git)
├── embedding_ready_reviews_small.parquet   # Default review source (your file may differ)
└── src/chinmay/                # Default META_CSV location
    └── Appliances_meta.csv
```

### Import flow (mental model)

1. **`app.py`** loads the chain via **`review_app.rag`**, runs the question form, then renders recommendations using **`review_app.data`** + **`review_app.ui`**.
2. **`review_app.rag`** owns LangChain **RetrievalQA**, **FAISS**, **HuggingFaceEmbeddings**, and **Ollama**.
3. **`review_app.data`** isolates pandas/IO and ranking; it uses Streamlit **`@st.cache_data`** for repeated loads.
4. **`review_app.ui`** holds all heavy Streamlit layout/HTML for cards and the product dialog.

---

## Environment variables (summary)

| Variable | Purpose |
|----------|---------|
| `REVIEWS_PARQUET` | Path to reviews parquet |
| `META_CSV` | Path to metadata CSV |
| `FAISS_INDEX_DIR` | Directory of saved FAISS index |
| `BUILD_FAISS_INDEX` | Set to `1` / `true` / `yes` to rebuild index once |

---

## Troubleshooting

- **`Could not load the RAG chain`**  
  Ensure **`ollama serve`** is running, **`ollama pull llama3`** finished, and **`faiss_index/`** exists.

- **Connection refused to Ollama**  
  Start the server: `ollama serve`, then retry.

- **First answer is slow**  
  The first Llama inference after startup is often much slower; later requests are faster while the model stays loaded.

- **ROUGE shows “—”**  
  Install `rouge-score` if missing: `pip install rouge-score`. ROUGE also needs a non-empty answer and retrieved documents.

---

## License

Not specified.
