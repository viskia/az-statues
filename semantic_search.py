import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_FILE = "cached_chunks.json"
VECTORS_FILE = "cached_vectors.npy"

print("Loading semantic search model (only happens once, at startup)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
chunk_metadata = []
chunk_embeddings = None


def _statutes_json_is_newer_than_cache():
    if not (os.path.exists(CHUNKS_FILE) and os.path.exists(VECTORS_FILE)):
        return True   # no cache yet, so "newer" by default -> rebuild
    statutes_time = os.path.getmtime("statutes.json")
    cache_time = os.path.getmtime(CHUNKS_FILE)
    return statutes_time > cache_time


def _build_and_cache():
    global chunks, chunk_metadata, chunk_embeddings

    print("Building semantic index from statutes.json (this may take a while)...")
    with open("statutes.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    chunks = []
    chunk_metadata = []

    for item in raw_data:
        paragraphs = item["text"].split("\n\n")
        for para in paragraphs:
            clean_para = para.strip()
            if not clean_para:
                continue
            contextual_chunk = f"ARS § {item['section']} ({item['heading']}): {clean_para}"
            chunks.append(contextual_chunk)
            chunk_metadata.append({
                "section": item["section"],
                "heading": item["heading"],
                "url": item["url"]
            })

    chunk_embeddings = model.encode(chunks, show_progress_bar=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadata": chunk_metadata}, f)
    np.save(VECTORS_FILE, chunk_embeddings)
    print("Semantic index built and cached.")


def _load_from_cache():
    global chunks, chunk_metadata, chunk_embeddings
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
        chunks = cache_data["chunks"]
        chunk_metadata = cache_data["metadata"]
    chunk_embeddings = np.load(VECTORS_FILE)
    print(f"Loaded {len(chunks)} cached semantic search blocks.")


# --- Run once, when this module is first imported ---
if _statutes_json_is_newer_than_cache():
    _build_and_cache()
else:
    _load_from_cache()


def semantic_search(query, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for index in top_indices:
        results.append({
            "score": float(similarities[index]),
            "section": chunk_metadata[index]["section"],
            "heading": chunk_metadata[index]["heading"],
            "url": chunk_metadata[index]["url"],
            "snippet": chunks[index]
        })
    return results