"""
Clear the ChromaDB collection to start fresh.
"""

import chromadb
from app.config import settings

client = chromadb.PersistentClient(path=settings.chroma_dir)
collection = client.get_or_create_collection(name=settings.collection_name)

count_before = collection.count()
print(f"Documents before clear: {count_before}")

if count_before > 0:
    # Get all IDs and delete them
    all_data = collection.get()
    if all_data["ids"]:
        collection.delete(ids=all_data["ids"])
        print(f"Deleted {len(all_data['ids'])} documents")

count_after = collection.count()
print(f"Documents after clear : {count_after}")

if count_after == 0:
    print("\n✓ Collection cleared successfully")
else:
    print(f"\n✗ Warning: {count_after} documents still remain")
