import json

with open("statutes.json") as f:
    statutes = json.load(f)

query = input("Search term: ").lower()
query_words = query.split()

results = []
for statute in statutes:
    combined_text = (statute["heading"] + " " + statute["text"]).lower()

    match_count = 0
    for word in query_words:
        if word in combined_text:
            match_count += 1

    if match_count > 0:
        results.append((match_count, statute))

results.sort(key=lambda pair: pair[0], reverse=True)

print(f"\nFound {len(results)} result(s):\n")
for match_count, r in results:
    print(f"§{r['section']} — {r['heading']}  ({match_count}/{len(query_words)} words matched)")

    # Find where the first query word appears in the actual text, and show context around it
    text_lower = r["text"].lower()
    first_word = query_words[0]
    position = text_lower.find(first_word)

    if position != -1:
        start = max(0, position - 60)
        end = position + 100
        snippet = r["text"][start:end].strip()
        print(f"  ...{snippet}...")

    print(f"  {r['url']}")
    print()