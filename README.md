# MkDocs-RAG-Assistant

A Retrieval-Augmented Generation (RAG) system for answering MkDocs documentation questions.
Uses cleaned and chunked MkDocs user guides, embedded with MPNet in a Chroma vector DB, and queried via a controlled System/Human prompt using Gemini. Includes CLI and Streamlit interfaces.

---

## 1. Project Overview

This project implements a RAG pipeline dedicated to the MkDocs documentation. It retrieves the most relevant documentation chunks using embeddings and vector search, then constructs a strict System/Human prompt to ensure the LLM answers only based on the documentation and resists prompt-injection attempts.

The system demonstrates:

* Selected chunking method and justification
* Markdown cleaning and preprocessing
* Text and image embedding model selection
* Vector database integration using Chroma
* Configurable top-k retrieval strategy (choice of k neighbours)
* Strict System / Human prompt separation
* CLI and Streamlit front-ends
* multimodal ingestion (text + image)

---

## 2. Repository Structure

```text
├── ingest.py               # Clean, chunk, embed, and store docs/images in Chroma
├── query_cli.py            # CLI RAG assistant (with configurable k neighbours)
├── app.py                  # Streamlit web-based RAG interface
├── requirements.txt
└── README.md
```

---

## 3. Data Source

All knowledge is extracted from the official MkDocs repository:

* Source: [https://github.com/mkdocs/mkdocs/tree/master](https://github.com/mkdocs/mkdocs/tree/master)
* Used subset: MkDocs **user guides** (`.md` files) copied under:

```text
data/mkdocs_docs/
```

Documentation images are stored in:

```text
data/mkdocs_images/
```

Only these files are used to build the knowledge base.

---

## 4. Installation and Setup

### 4.1 Requirements

* Python 3.9 or later
* A valid Google Gemini API key

### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Configure Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

---

## 5. Ingestion Pipeline (`ingest.py`)

### 5.1 Chunking Method and Rationale

Text is split into overlapping character chunks using a sliding window:

```python
CHUNK_SIZE = 700       # characters
CHUNK_OVERLAP = 150    # characters
```

**Rationale**

* Keeps related sentences and code snippets in the same chunk.
* Overlap preserves context across chunk boundaries.
* Chunks are long enough to be informative but short enough for efficient retrieval and prompting.

This method gives good recall while limiting noise during top-k retrieval.

### 5.2 Markdown Cleaning

Before chunking, each markdown document is processed by `clean_markdown(text)` to keep only readable text.

Cleaning steps:

* Remove fenced code block markers (``` and language hints) but keep the code content.
* Remove inline backticks: `` `code` `` → `code`.
* Convert links: `[text](url)` → `text`.
* Convert images: `![alt](url)` → `alt`.
* Remove MkDocs anchors: `{#anchor-id}`.
* Strip heading markers (`#`, `##`, `###`, …) while keeping the heading text.
* Collapse multiple whitespace characters and trim the result.

The output is compact, semantic text suitable for embedding and retrieval.

### 5.3 Embedding Models

Two separate embedding models are used.

#### 5.3.1 Text Embedding Model

```python
TEXT_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
```

Chosen because:

* Strong performance on semantic similarity and Q&A tasks.
* Works well with technical documentation.
* Widely used and stable for RAG systems.

All cleaned text chunks are embedded with this model and stored in the text collection.

#### 5.3.2 Image Embedding Model (Multimodal)

```python
IMAGE_EMBED_MODEL_NAME = "clip-ViT-B-32"
```

Images under `data/mkdocs_images/` are embedded using CLIP and stored in a separate image collection. This enables multimodal retrieval (text queries can later be matched to images).

### 5.4 Vector Database (Chroma)

The project uses **Chroma** as the vector database:

```python
client = chromadb.PersistentClient(
    path=str(DB_DIR),
    settings=Settings(anonymized_telemetry=False),
)
```

Two persistent collections are created:

* `mkdocs_text` – text chunks, embeddings, and metadata (source file path, chunk index, type).
* `mkdocs_images` – image embeddings and metadata (image path, type).

Ingestion is started with:

```bash
python ingest.py
```

This step only needs to be run once (or after updating the docs).

---

## 6. Retrieval and RAG Pipeline (`query_cli.py`)

### 6.1 Retrieval Flow

The main RAG flow is implemented in `ask_mkdocs()` and `ask_mkdocs_with_k()`:

1. Encode the user question using `all-mpnet-base-v2`.
2. Query the `mkdocs_text` Chroma collection with `n_results = k`.
3. Build a Human prompt that includes the retrieved context chunks and the question.
4. Provide a separate System prompt that defines the assistant’s behaviour.
5. Call Gemini (`gemini-2.0-flash`) with these prompts.
6. Return the generated answer and the list of context chunks used.

### 6.2 Selecting the Number of Neighbours *k*

A key design choice is the number of neighbours *k* used during retrieval.

* Default value:

  ```python
  K_NEIGHBOURS = 5
  ```

* In the CLI, the user can override this per question:

  ```text
  How many neighbours (k) to use? [default = 5]:
  ```

**Reasoning**

* Small `k` (2–3): very focused context but may miss complementary details.
* Medium `k` (around 5): good trade-off between completeness and noise.
* Larger `k` (> 7): adds redundant or less relevant text and increases prompt length.

Experiments showed that `k = 5` gives clear, well-grounded answers while keeping the prompt compact.
<img width="1613" height="215" alt="image" src="https://github.com/user-attachments/assets/2180df76-e72a-4834-a0f3-f13db8a936e8" />
<img width="632" height="285" alt="image" src="https://github.com/user-attachments/assets/d35a5e8d-f340-47f0-b64e-7d4b5c8709b0" />
<img width="1242" height="166" alt="image" src="https://github.com/user-attachments/assets/3ed16e29-b0c6-46e4-988f-9c97c50e136b" />
<img width="1610" height="514" alt="image" src="https://github.com/user-attachments/assets/69ff24bd-0d23-4ecf-92db-79684861f473" />









### 6.3 System / Human Prompt Separation

To ensure the model only answers MkDocs-related questions and ignores attempts to change its behaviour, the prompt is split into:

* **System prompt** (fixed rules)

  * Defines the assistant as a MkDocs helper.
  * Instructs it to rely primarily on the provided `<context>`.
  * Forbids inventing new features or commands not present in the context.
  * Tells the model to ignore user requests that conflict with these rules.

* **Human prompt** (dynamic content)

  * Contains the retrieved context chunks, wrapped in a `<context>` block.
  * Includes the user’s question.
  * Asks the model to answer based on the context and to state clearly when information is missing.

This separation makes the system more robust against prompt-injection and keeps it aligned with the assignment’s requirement that the LLM should not answer outside its use case.

---

## 7. CLI Usage

Run the command-line assistant:

```bash
python query_cli.py
```

Example interaction:

```text
MkDocs RAG assistant (CLI)
Type 'exit' to quit.

Ask about MkDocs: How do I install MkDocs?
How many neighbours (k) to use? [default = 5]: 3
```

The program prints:

* `=== Answer ===` – the generated answer.
* `=== Context chunks used ===` – the retrieved chunks (shortened for readability).

This makes it easy to see how retrieval affects the final answer.

---

## 8. Streamlit Web Interface

Run the web interface:

```bash
streamlit run app.py
```

Features:

* Input box to ask MkDocs questions.
* Chat-like layout showing your questions and the assistant’s answers.
* Expandable section for each answer showing the context chunks that were used.

The Streamlit app reuses the same RAG pipeline (via `ask_mkdocs()`), so behaviour is consistent with the CLI.

---

## 9. Example Question and Context

Example question used for testing:

> **How do I install MkDocs?**

Typical retrieved chunks (with `k = 4` or `5`) include:

* Requirements for Python and `pip`.
* The command: `pip install mkdocs`.
* Notes about upgrading `pip` or checking the Python version.
* Instructions to verify the installation with `mkdocs --version`.

The answer is constructed only from these chunks and explicitly follows the documentation.
<img width="1746" height="773" alt="image" src="https://github.com/user-attachments/assets/aec599b9-e8d7-4fc7-bbf9-4103719d5752" />




---

## 10. Start Summary

1. Download MkDocs user-guide markdown files into `data/mkdocs_docs/`.
2. Create `.env` with your `GEMINI_API_KEY`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run `python ingest.py` to build the Chroma DB.
5. Use `python query_cli.py` for CLI interaction or `streamlit run app.py` for the web UI.
