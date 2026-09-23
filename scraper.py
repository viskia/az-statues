import cloudscraper
from bs4 import BeautifulSoup
import time
import json
import os

scraper = cloudscraper.create_scraper()


def get_section_urls(title_num):
    list_url = f"https://www.azleg.gov/arsDetail/?title={title_num}"
    response = scraper.get(list_url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", class_="stat")
    return [link["href"].split("docName=")[1] for link in links]


def scrape_title(title_num, existing_urls):
    section_urls = get_section_urls(title_num)
    print(f"Title {title_num}: found {len(section_urls)} sections.")

    new_sections = []
    skipped = []

    for i, url in enumerate(section_urls, start=1):
        if url in existing_urls:
            continue   # already scraped this one, skip it (resume-safe)

        print(f"[{i}/{len(section_urls)}] {url}")
        response = scraper.get(url)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")

        try:
            section_number = paragraphs[0].find("font", color="GREEN").get_text().strip(".")
            heading = paragraphs[0].find("font", color="PURPLE").get_text().strip()
            body_text = ""
            for p in paragraphs[1:]:
                body_text += p.get_text() + "\n\n"

            new_sections.append({
                "title": str(title_num),
                "section": section_number,
                "heading": heading,
                "text": body_text.strip(),
                "url": url
            })
        except (IndexError, AttributeError):
            skipped.append(url)

        time.sleep(0.5)

    return new_sections, skipped


# --- Load whatever already exists, so we never lose it ---
if os.path.exists("statutes.json"):
    with open("statutes.json") as f:
        all_statutes = json.load(f)
else:
    all_statutes = []

existing_urls = {s["url"] for s in all_statutes}

# --- Scrape a new title ---
TITLE_TO_SCRAPE = 28   # change this number each time you scrape a new title

new_sections, skipped = scrape_title(TITLE_TO_SCRAPE, existing_urls)
all_statutes.extend(new_sections)

with open("statutes.json", "w") as f:
    json.dump(all_statutes, f, indent=2)

print(f"\nAdded {len(new_sections)} new sections from Title {TITLE_TO_SCRAPE}.")
print(f"Skipped {len(skipped)} (unexpected format).")
print(f"Total statutes in file now: {len(all_statutes)}")