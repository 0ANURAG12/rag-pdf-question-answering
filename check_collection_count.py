"""
Check the current ChromaDB collection count and show breakdown by source.
Run this after each API upload to verify idempotency.
"""

import chromadb
from collections import Counter
from app.config import settings

client = chromadb.PersistentClient(path=settings.chroma_dir)
collection = client.get_or_create_collection(name=settings.collection_name)

total = collection.count()
print(f"Total documents in collection: {total}")

if total > 0:
    all_data = collection.get(include=["metadatas"])
    sources = [m["source"] for m in all_data["metadatas"]]
    source_counts = Counter(sources)
    
    print("\nBreakdown by source:")
    for source, count in source_counts.items():
        print(f"  - {source}: {count} chunks")
else:
    print("Collection is empty")
