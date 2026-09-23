import cloudscraper
from bs4 import BeautifulSoup
import json

scraper = cloudscraper.create_scraper()


def scrape_chapters(title_num):
    list_url = f"https://www.azleg.gov/arsDetail/?title={title_num}"
    response = scraper.get(list_url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    chapter_links = soup.find_all("a", class_="one-sixth first")
    chapters = []
    for link in chapter_links:
        chapter_text = link.get_text().strip()
        if not chapter_text.startswith("Chapter"):
            continue
        name_div = link.find_next_sibling("div", class_="two-thirds")
        range_div = link.find_next_sibling("div", class_="one-sixth")
        chapters.append({
            "chapter": chapter_text,
            "name": name_div.get_text().strip() if name_div else "",
            "range": range_div.get_text().replace("Sec:", "").strip() if range_div else ""
        })
    return chapters


def parse_section_number(section_str):
    parts = section_str.split("-")
    return float(parts[1])


def parse_chapter_range(range_str):
    parts = range_str.split("-")
    start = float(parts[1])
    end = float(parts[3])
    return start, end


def assign_chapters_for_title(statutes, title_num, chapters):
    title_str = str(title_num)
    for statute in statutes:
        if statute.get("title") != title_str:
            continue   # only touch statutes belonging to this title
        if statute.get("chapter"):
            continue   # already tagged, don't redo work

        statute["chapter"] = "Uncategorized"
        try:
            statute_num = parse_section_number(statute["section"])
        except (IndexError, ValueError):
            continue

        for chapter in chapters:
            try:
                start, end = parse_chapter_range(chapter["range"])
            except (IndexError, ValueError):
                continue
            if start <= statute_num <= end:
                statute["chapter"] = f"{chapter['chapter']} — {chapter['name'].title()}"
                break

    return statutes


TITLE_TO_TAG = 15 # match whatever you just scraped

with open("statutes.json") as f:
    statutes = json.load(f)

chapters = scrape_chapters(TITLE_TO_TAG)
statutes = assign_chapters_for_title(statutes, TITLE_TO_TAG, chapters)

with open("statutes.json", "w") as f:
    json.dump(statutes, f, indent=2)

print(f"Done tagging Title {TITLE_TO_TAG}.")