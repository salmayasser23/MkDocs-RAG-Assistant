"""
Ingest MkDocs documentation into a Chroma vector DB.

- Cleans markdown text.
- Chunks text with a sliding window.
- Uses a TEXT embedding model for docs (good for Q&A).
- Uses CLIP for IMAGE embeddings.
- Stores text and images in two persistent Chroma collections:
  - "mkdocs_text"   (text chunks)
  - "mkdocs_images" (image embeddings)
"""

import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image

# Paths & constants

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "mkdocs_docs"
IMAGES_DIR = BASE_DIR / "data" / "mkdocs_images"
DB_DIR = BASE_DIR / "chroma_db"

# Two separate collections: one for text, one for images
TEXT_COLLECTION_NAME = "mkdocs_text"
IMAGE_COLLECTION_NAME = "mkdocs_images"

# Text model: good semantic search for RAG
TEXT_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Image model: CLIP for multimodal bonus
IMAGE_EMBED_MODEL_NAME = "clip-ViT-B-32"

CHUNK_SIZE = 700       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks


# Helper functions

def clean_markdown(text: str) -> str:
    """Basic markdown cleaning to keep only readable text."""
    # Keep fenced code blocks content: remove ``` but keep inner text
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL)

    # Remove inline code backticks but keep content
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Images: ![alt](url) -> alt text
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # MkDocs anchors {#id}
    text = re.sub(r"{#.*?}", " ", text)

    # Headings (#, ##, ...)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP):
    """Split long text into overlapping character chunks."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == length:
            break

        # move window with overlap
        start = end - overlap

    return chunks


# ----------------- Ingestion functions -----------------

def ingest_text_docs(model: SentenceTransformer, collection):
    """Read, clean, chunk and store all markdown files in the text collection."""
    md_files = sorted(DOCS_DIR.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"No .md files found in {DOCS_DIR}")

    ids = []
    documents = []
    metadatas = []

    print(f"Found {len(md_files)} markdown files.")
    print("Cleaning, chunking and collecting text...")

    for md_path in tqdm(md_files):
        raw = md_path.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw)
        chunks = chunk_text(cleaned)

        # use full relative path to guarantee unique IDs
        rel_path = md_path.relative_to(BASE_DIR).as_posix()

        for idx, chunk in enumerate(chunks):
            doc_id = f"{rel_path}_chunk_{idx}"

            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": rel_path,
                    "chunk_index": idx,
                    "type": "text",
                }
            )

    if not documents:
        raise RuntimeError("No non-empty chunks were generated from docs.")

    print(f"Embedding {len(documents)} text chunks...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # All lists must have the same length
    assert len(ids) == len(documents) == len(metadatas) == len(embeddings)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("Text documents ingested successfully.")


def ingest_images(model: SentenceTransformer, collection):
    """Embed images into the image collection"""
    if not IMAGES_DIR.exists():
        print("Images directory not found, skipping images.")
        return

    img_paths = [
        p for p in IMAGES_DIR.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    ]
    if not img_paths:
        print("No images found, skipping images.")
        return

    print(f"Embedding {len(img_paths)} images...")

    ids = []
    embeddings = []
    metadatas = []

    for img_path in tqdm(img_paths):
        img = Image.open(img_path).convert("RGB")
        emb = model.encode(img).tolist()

        rel_img = img_path.relative_to(BASE_DIR).as_posix()
        ids.append(f"image_{rel_img}")
        embeddings.append(emb)
        metadatas.append(
            {
                "source": rel_img,
                "type": "image",
            }
        )

    assert len(ids) == len(embeddings) == len(metadatas)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Images ingested successfully.")


# ----------------- Main entry point -----------------

def main():
    print(f"Docs directory:   {DOCS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")

    DB_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Two separate collections: text and images
    text_collection = client.get_or_create_collection(
        name=TEXT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    image_collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Using TEXT embedding model:  {TEXT_EMBED_MODEL_NAME}")
    text_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)

    print(f"Using IMAGE embedding model: {IMAGE_EMBED_MODEL_NAME}")
    image_model = SentenceTransformer(IMAGE_EMBED_MODEL_NAME)

    # Ingest docs into text collection
    ingest_text_docs(text_model, text_collection)

    # Ingest images into image collection (optional bonus)
    ingest_images(image_model, image_collection)

    print("=== Ingestion finished successfully ===")


if __name__ == "__main__":
    main()
