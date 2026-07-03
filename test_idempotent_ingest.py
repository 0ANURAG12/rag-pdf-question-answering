"""
Idempotency test: re-ingest the same PDF three times and verify
that the ChromaDB document count stays constant.
"""

import sys
import chromadb
from app.config import settings
from app.services import pdf_service, chunking_service, embedding_service, vectorstore_service

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if not PDF_PATH:
    print("Usage: python test_idempotent_ingest.py <path_to_pdf>")
    sys.exit(1)

print("=" * 70)
print("IDEMPOTENCY TEST")
print("=" * 70)

# Read the PDF once
with open(PDF_PATH, "rb") as f:
    file_bytes = f.read()

filename = PDF_PATH.split("/")[-1]
text = pdf_service.extract_text(file_bytes)
chunks = chunking_service.split_text(text)
embeddings = embedding_service.embed_texts(chunks)

print(f"\nPDF          : {filename}")
print(f"Chunks per run: {len(chunks)}")
print()

# Connect directly to check counts
client = chromadb.PersistentClient(path=settings.chroma_dir)
collection = client.get_or_create_collection(name=settings.collection_name)

count_before = collection.count()
print(f"Count before any ingestion : {count_before}")

for run in range(1, 4):
    vectorstore_service.add_chunks(chunks, embeddings, source=filename)
    count = collection.count()
    print(f"Count after ingestion #{run}   : {count}")

print()
final_count = collection.count()
expected = len(chunks)

if final_count == expected:
    print(f"✓ PASS — count is stable at {final_count} (matches {expected} chunks per ingestion)")
else:
    print(f"✗ FAIL — expected {expected}, got {final_count}")

print("=" * 70)
