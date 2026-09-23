import json

with open("statutes.json") as f:
    statutes = json.load(f)

fixed_count = 0
for statute in statutes:
    if not statute.get("title"):
        # Pull the title number from the section string, e.g. "13-1402" -> "13"
        title_num = statute["section"].split("-")[0]
        statute["title"] = title_num
        fixed_count += 1

with open("statutes.json", "w") as f:
    json.dump(statutes, f, indent=2)

print(f"Backfilled 'title' field on {fixed_count} statutes.")