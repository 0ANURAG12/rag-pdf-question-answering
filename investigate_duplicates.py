"""
Diagnostic script to investigate duplicate retrieval root cause.
"""

import chromadb
from collections import Counter

# Connect to the same ChromaDB instance
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="documents")

print("=" * 80)
print("CHROMADB DUPLICATE INVESTIGATION")
print("=" * 80)

# 1. Total document count
total_count = collection.count()
print(f"\n1. Total documents in ChromaDB collection: {total_count}")

# 2. Get all documents
if total_count > 0:
    all_data = collection.get(
        include=["metadatas", "documents", "embeddings"]
    )
    
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    documents = all_data["documents"]
    
    print(f"\n2. Total unique IDs: {len(set(ids))}")
    print(f"   Total IDs: {len(ids)}")
    
    if len(ids) != len(set(ids)):
        print("   ⚠️  DUPLICATE IDs FOUND!")
        id_counts = Counter(ids)
        duplicates = {id_: count for id_, count in id_counts.items() if count > 1}
        print(f"   Duplicate IDs: {duplicates}")
    else:
        print("   ✓ All IDs are unique")
    
    # 3. Check sources
    sources = [m["source"] for m in metadatas]
    source_counts = Counter(sources)
    print(f"\n3. Sources found:")
    for source, count in source_counts.items():
        print(f"   - {source}: {count} chunks")
    
    # 4. Check chunk_index per source
    print(f"\n4. Chunk index analysis per source:")
    for source in source_counts.keys():
        source_chunks = [m for m in metadatas if m["source"] == source]
        chunk_indices = [m["chunk_index"] for m in source_chunks]
        unique_indices = set(chunk_indices)
        
        print(f"\n   Source: {source}")
        print(f"   - Total chunks: {len(chunk_indices)}")
        print(f"   - Unique chunk_index values: {len(unique_indices)}")
        print(f"   - Chunk_index range: {min(chunk_indices)} to {max(chunk_indices)}")
        
        if len(chunk_indices) != len(unique_indices):
            print("   ⚠️  DUPLICATE chunk_index values found!")
            index_counts = Counter(chunk_indices)
            duplicates = {idx: count for idx, count in index_counts.items() if count > 1}
            print(f"   Duplicate chunk_index values: {duplicates}")
        else:
            print("   ✓ All chunk_index values are unique for this source")
    
    # 5. Check for duplicate content
    print(f"\n5. Content duplication analysis:")
    content_counts = Counter(documents)
    duplicate_content = {content[:100]: count for content, count in content_counts.items() if count > 1}
    
    if duplicate_content:
        print(f"   ⚠️  {len(duplicate_content)} unique content pieces appear multiple times:")
        for content_preview, count in list(duplicate_content.items())[:5]:
            print(f"   - '{content_preview}...' appears {count} times")
    else:
        print("   ✓ All content is unique")
    
    # 6. Check (source, chunk_index) pairs
    print(f"\n6. (source, chunk_index) pair uniqueness:")
    pairs = [(m["source"], m["chunk_index"]) for m in metadatas]
    unique_pairs = set(pairs)
    
    print(f"   - Total pairs: {len(pairs)}")
    print(f"   - Unique pairs: {len(unique_pairs)}")
    
    if len(pairs) != len(unique_pairs):
        print("   ⚠️  DUPLICATE (source, chunk_index) pairs found!")
        pair_counts = Counter(pairs)
        duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
        print(f"   Duplicate pairs (showing first 10):")
        for pair, count in list(duplicates.items())[:10]:
            print(f"   - {pair}: {count} occurrences")
    else:
        print("   ✓ All (source, chunk_index) pairs are unique")
    
    # 7. Sample query to check retrieval
    print(f"\n7. Sample query test:")
    sample_embedding = all_data["embeddings"][0] if all_data["embeddings"] else None
    
    if sample_embedding:
        query_results = collection.query(
            query_embeddings=[sample_embedding],
            n_results=min(10, total_count)
        )
        
        returned_ids = query_results["ids"][0]
        returned_metadatas = query_results["metadatas"][0]
        
        print(f"   - Query requested: 10 results")
        print(f"   - Query returned: {len(returned_ids)} results")
        
        # Check for duplicate chunk_index in results
        returned_pairs = [(m["source"], m["chunk_index"]) for m in returned_metadatas]
        unique_returned = set(returned_pairs)
        
        if len(returned_pairs) != len(unique_returned):
            print("   ⚠️  Query returned duplicate (source, chunk_index) pairs!")
            pair_counts = Counter(returned_pairs)
            duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
            print(f"   Duplicate pairs in results: {duplicates}")
        else:
            print("   ✓ Query results contain unique (source, chunk_index) pairs")

else:
    print("\n❌ Collection is empty - no data to analyze")

print("\n" + "=" * 80)
