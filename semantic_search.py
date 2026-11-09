import json
import os
import re # regular expressions, using to find patterns in text to fix chunking
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_FILE = "cached_chunks.json"
VECTORS_FILE = "cached_vectors.npy"

print("Loading semantic search model (only happens once, at startup)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# how does SentenceTransformer work?
# it converts text into a vector representation that captures semantic meaning, so similar texts have similar vectors.
# we're using it to read the query & statute separately, turn each one into a list of numbers (a vector), and then
# compare the vectors to find statutes that are semantically similar to the query.

chunks = []
chunk_metadata = []
chunk_embeddings = None


def _statutes_json_is_newer_than_cache():
    if not (os.path.exists(CHUNKS_FILE) and os.path.exists(VECTORS_FILE)):
        return True   # no cache yet, so "newer" by default -> rebuild
    statutes_time = os.path.getmtime("statutes.json")
    cache_time = os.path.getmtime(CHUNKS_FILE)
    return statutes_time > cache_time

MAX_CHUNK_WORDS = 150 # the embedding model reads ~256 tokens, legal text is abt ~1.3 tokens per word, + the ARS § prefix, so 150 words keeps chunks within the model's limit

# this is used to determine the nesting level of statute list items, using it to learn how to use regular expressions (re)
def item_level(paragraph):
    """how deeply indented/nested a statute list item is: ex. 1 for "A.", 2 for "1.", 3 for "(a.) none if its not a list item"""
    # the triple """ marks the start and end of a docstring, which are the standard way to document what a function does. they go on the line right after def
    if re.match(r"^[A-Z]\.\s", paragraph): # [A-Z] means any capital letter, + \. means any literal period, + \s means any whitespace e.g. "A. a person..."
        return 1
    if re.match(r"^\d+\.\s", paragraph): # \d+ means one or more digits + a any literal period + any whitespace e.g. "1. ..."
        return 2
    if re.match(r"^\([a-z0-9]+\)", paragraph): # \( means an opening parenthesis, [a-z0-9]+ means one or more lowercase letters or digits, and \) means a closing parenthesis. e.g. "(a) ..."
        return 3
    return None # means the paragraph is not a list item, just ordinary text

def group_paragraphs(paragraphs):
    """keeps each list item together with the items nested under it, 
    e.g. ["A. ... by:", "1. ...", "2. ...", "B. ..."] -> [["A. ... by:", "1. ...", "2. ..."], ["B. ..."]]"""
    groups = [] # each inner list contains paragraphs grouped together
    group_level = None # the indentation level of the paragraph that started the current group

    for para in paragraphs:
        level = item_level(para)
        # if a group exists, the current paragraph is a nested list item, 
        # and it is nested deeper (bigger level number) than the current group's first paragraph, add it to the current group
        if groups and level is not None and group_level is not None and level > group_level:
            groups[-1].append(para) # [-1] = the last (current) group
        else:
            groups.append([para]) # start a new group containing just this paragraph
            group_level = level
    return groups

def count_words(paragraphs):
    """counts the total number of words in a list of paragraphs"""
    return sum(len(p.split()) for p in paragraphs)

def split_group(group):
    """splits a group into chunks of at most MAX_CHUNK_WORDS words, 
    each piece starts with the group's first line so it never loses its context"""
    lead = group[0]
    pieces = []
    current = [lead]

    for para in group[1:]:
        too_long = count_words(current + [para]) > MAX_CHUNK_WORDS
        if too_long and len(current) > 1:
            pieces.append(current)
            current = [lead, para]
        else:
            current.append(para)
    pieces.append(current)
    return pieces

def _build_and_cache():
    global chunks, chunk_metadata, chunk_embeddings

    print("Building semantic index from statutes.json (this may take a while)...")
    with open("statutes.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    chunks = []
    chunk_metadata = []

    for item in raw_data:
            paragraphs = [p.strip() for p in item["text"].split("\n\n") if p.strip()]

            for group in group_paragraphs(paragraphs):
                for piece in split_group(group):
                    body = " ".join(piece)
                    contextual_chunk = f"ARS § {item['section']} ({item['heading']}): {body}"
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

chunk_sections = np.array([m["section"] for m in chunk_metadata])

def semantic_search(query, top_k=5, allowed_sections=None):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    if allowed_sections is not None:
        mask = np.isin(chunk_sections, list(allowed_sections))
        similarities = np.where(mask, similarities, -np.inf)

    sorted_indices = np.argsort(similarities)[::-1] # ALL chunks, but the best first

    results = []
    seen_sections = set()

    for index in sorted_indices:
        score = similarities[index]
        if score == -np.inf:
            break # everything after this is outside the filter

        section = chunk_metadata[index]["section"]
        if section in seen_sections:
            continue # we already have a better paragraph from this statute
        seen_sections.add(section)

        results.append({
            "score": float(similarities[index]),
            "section": chunk_metadata[index]["section"],
            "heading": chunk_metadata[index]["heading"],
            "url": chunk_metadata[index]["url"],
            "snippet": chunks[index], # the paragraph that matched the query best, re-ranker will read instead of first 500 characters
        })

        if len(results) == top_k:
            break # got enough diff statutes, stop

    return results