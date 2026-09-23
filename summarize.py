from groq import Groq
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open("statutes.json") as f:
    statutes = json.load(f)

for i, statute in enumerate(statutes, start=1):
    if statute.get("summary"):
        continue

    print(f"[{i}/{len(statutes)}] Summarizing {statute['section']}...")

    for attempt in range(3):   # try up to 3 times
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Summarize this Arizona statute in ONE short sentence (under 20 words). "
                        f"Plain English, no legal jargon, no preamble like 'This statute...'. "
                        f"Just state what it does.\n\n"
                        f"{statute['text']}"
                    )
                }]
            )
            statute["summary"] = response.choices[0].message.content.strip()
            break   # success, stop retrying

        except Exception as e:
            print(f"  -> Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(10)   # wait longer before retrying
            else:
                statute["summary"] = None   # give up after 3 tries

    time.sleep(1.5)

    if i % 25 == 0:
        with open("statutes.json", "w") as f:
            json.dump(statutes, f, indent=2)
        print(f"  (progress saved at {i})")

with open("statutes.json", "w") as f:
    json.dump(statutes, f, indent=2)

print("Done.")