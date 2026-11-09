import os

from flask import Flask, request
from markupsafe import escape
import json
import re
import math
from semantic_search import semantic_search
from sentence_transformers import CrossEncoder
# A note on how CrossEncoder works because I don't know what I'm doing:
# it reads the search query and a piece of the statute text together, comparing each word, and outputs a relevance score. 
# More accurate than most other steps, but it is slower because it scores each query-document pair individually & nothing can be precomputed.
# this is the standard design for search systems:
# 1. RETRIEVE candidate documents using a fast method (semantic search/sentence transformer)
# 2. RE-RANK the candidates using a more accurate model (cross-encoder)

app = Flask(__name__)

with open("statutes.json") as f:
    statutes = json.load(f)

statutes_by_section = {s["section"]: s for s in statutes} # dictionary comprehension; builds dict mapping each section # to its statute

print("Loading re-ranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   

TITLE_NAMES = {
    #"3": "Agriculture",
    "4": "Alcoholic Beverages and Alternative Nicotine Products",
    #"5": "Amusements and Sports", 
    #"6": "Banks and Financial Institutions",
    #"7": "Bonds",
    #"8": "Child Safety",
    #"9": "Cities and Towns",
    #"10": "Corporations and Associations",
    #"11": "Counties",
    #"12": "Courts and Civil Proceedings",
    "13": "Criminal Code",
    #"14": "Trusts, Estates, and Protecting Proceedings"
    "15": "Education",
    #"16": "Elections and Electors",
    #"17": "Game and Fish",
    #"18": "Information Technology",
    #"19": "Initiative, Referendum and Recall",
    #"20": "Insurance",
    #"21": "Juries", 
    #"22": "Justice and Municipal Courts",
    #"23": "Labor",
    #"25": "Marital and Domestic Relations",
    #"26": "Military Affairs and Emergency Management",
    #"27": "Minerals, Oil and Gas",
    "28": "Transportation",
    #"29": "Partnership", 
    #"30": "Power",
    "31": "Prisons and Prisoners",
    #"32": "Professions and Occupations":
    #"33": "Property":
    #"34": "Public Buildings and Improvements",
    #"35": "Public Finances":
    "36": "Public Health and Safety",
    #"37": "Public Lands",
    #"38": "Public Officers and Employees",
    #"39": "Public Records, Printing, and Notices",
    #"40": "Public Utilities and Carriers",
    #"41": "State Government",
    #"42": "Taxation":
    #"43": "Taxation of Income",
    #"44": "Trade and Commerce",
    #"45": "Waters",
    #"46": "Welfare",
    #"47": "Uniform Commerical Code",
    #"48": "Special Taxing Districts",
    #"49": "The Environment",
    # add an entry here each time you scrape a new title
}

def get_all_titles(statutes):
    titles = set()
    for statute in statutes:
        if statute.get("title"):
            titles.add(statute["title"])
    return sorted(titles, key=lambda t: int(t))

def get_chapters_for_title(statutes, title_num):
    chapters = set()
    for statute in statutes:
        if statute.get("title") == title_num and statute.get("chapter"):
            chapters.add(statute["chapter"])
    return sorted(chapters)

def build_document_frequency(statutes): # runs at startup, for every word counts how many statutes contain it at least once (document frequency)
    #ex. "the" might be in 5900 statutes and "shoplifting" might be in 2. it stores the result in a dictionary (doc_freq)
    doc_freq = {}
    for statute in statutes:
        text = (statute["heading"] + " " + statute["text"]).lower()
        words_in_this_statute = set(re.findall(r"\b\w+\b", text))
        for word in words_in_this_statute:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    return doc_freq

doc_freq = build_document_frequency(statutes)   # compute this once, when the app starts

def word_weight(word): #turns that docfreq into an important score/weight using inverse document frequency (IDF)
    freq = doc_freq.get(word, 1)
    return math.log(len(statutes) / freq)   # rarer words get a bigger number

def count_word_occurrences(word, text):
    """counts how many times a word appears in a piece of text using a regex"""
    pattern = r"\b" + re.escape(word) + r"\b" # \b means "word boundary" -- so \bcat\b matches "cat" but not "category"
    matches = re.findall(pattern, text)
    return len(matches)

def score_statute(statute, query_words): 
    """scores a statute based on how many query words it contains, weighting heading matches more than body matches"""
    heading_lower = statute["heading"].lower()
    text_lower = statute["text"].lower()

    score = 0
    for word in query_words:
        weight = word_weight(word)
        heading_hits = count_word_occurrences(word, heading_lower)
        body_hits = count_word_occurrences(word, text_lower)

        score += heading_hits * 10 * weight   # heading matches count much more
        score += body_hits * 1 * weight      # body matches count normally

    return score

STOPWORDS = {"of", "the", "a", "an", "in", "on", "and", "or", "to", "for", "is", "are"} # common words that don't add to search results

def search_statutes(query, pool):
    """splits the query into words, removes stopwords, scores every statute in the pool, keeps the ones scoring above 0, and sorts them best first."""
    query_words = query.lower().split()
    query_words = [w for w in query_words if w not in STOPWORDS]
    results = []

    for statute in pool:  # was: for statutes in statutes
        score = score_statute(statute, query_words)
        if score > 0:
            results.append((score, statute))

    results.sort(key=lambda pair: pair[0], reverse=True)
    return results

def get_snippet(statute, query_words): # get a snippet of the statute text around the first query word found, to show where keywords were found in the text
    text_lower = statute["text"].lower()
    for word in query_words:
        position = text_lower.find(word)
        if position != -1:
            start = max(0, position - 60)
            end = position + 100
            return statute["text"][start:end].strip()
    return statute["text"][:100].strip()

def highlight(text, query_words):
    escaped_text = str(escape(text))   # escape HTML first, so we don't break on <, >, & etc.
    for word in query_words:
        pattern = r"\b(" + re.escape(word) + r")\b"
        escaped_text = re.sub(pattern, r"<mark>\1</mark>", escaped_text, flags=re.IGNORECASE)
    return escaped_text

MIN_SEMANTIC_SCORE = 0.24   # tune results
RRF_K = 60 # standard RRF damping constant. note: RRF means Reciprocal Rank Fusion
RRF_LIST_DEPTH = 150 # maximum number of results to consider from each list for RRF

# re-ranker/crossencoder scores: positive = probably relevant, very negative = probably slop
# the scores aren't on a fixed scale, so a good match for one query can score differently than a good match for another,
# so these values are intentionally loose and are just rough guidelines chosen by testing 10 real and slop queries.
RERANK_NO_MATCH = -2.0 # if even the BEST result scores below this, treat the search as "no strong matches"
RERANK_MIN_SCORE = -5.0 # only keep results above this score, everything below is noise

def hybrid_search(query, top_n=200, title=None, chapter=None):
    pool = statutes
    if title:
        pool = [s for s in pool if s.get("title") == title]
    if chapter:
        pool = [s for s in pool if s.get("chapter") == chapter]
    allowed = {s["section"] for s in pool} if (title or chapter) else None

    keyword_results = search_statutes(query, pool)
    semantic_results = semantic_search(query, top_k=150, allowed_sections=allowed)

    # major update in how semantic & keyword results are combined, switched to RRF to prevent keyword outliers from dominating and ensure both have the same influence. Before I just averaged the two scores.

    fused_scores = {}
    # keyword results, each statute gets 1/(k + rank) points
    for rank, (score, statute) in enumerate(keyword_results[:RRF_LIST_DEPTH], start=1): # tuple unpacking; each keyword is a pair (score, statute) split into 2 variables, score isnt used
        section = statute["section"]       # keyword_results[:RRF_LIST_DEPTH] gives both lists equal weight, 150 votes each
        fused_scores[section] = fused_scores.get(section, 0) + 1 / (RRF_K + rank)

    # semantic results, each statute gets 1/(k + rank) points, adds to the keyword points
    best_snippets = {} # section number -> best semantic snippet
    for rank, r in enumerate(semantic_results, start=1):
        if r["score"] < MIN_SEMANTIC_SCORE:
            break # the list is already sorted by score so everything after this is too weak, stops at the first score that fails conditional
        section = r["section"]
        fused_scores[section] = fused_scores.get(section, 0) + 1 / (RRF_K + rank)
        best_snippets[section] = r["snippet"]
        # semantic_search returns one result per section
        # and it's that section's best paragraph so dont need to compare

    combined = [(score, statutes_by_section[section]) for section, score in fused_scores.items()]
    combined.sort(key=lambda pair: pair[0], reverse=True)
    # Return the ranked statutes and the best paragraph for each one, snippets travel in
    # their own dict instead of being written onto the statute dicts,
    # b/c 'statutes' is shared by every search, changing it would affect other searches
    return combined[:top_n], best_snippets # returns both values as a pair, & the caller can split them into two variables

def rerank(query, candidates, best_snippets, top_n=15):   
    pairs = []
    for score, r in candidates:
        # use the best-matching paragraph; keyword-only results have no snippet, so fall back to heading + start of text
        text = best_snippets.get(r["section"], f"{r['heading']}. {r['text'][:500]}")
        pairs.append((query, text))

    rerank_scores = reranker.predict(pairs)
    reranked = list(zip(rerank_scores, [r for score, r in candidates]))
    # zip() pairs up two lists item by item: (score1, statute1), (score2, statute2), ...
    reranked.sort(key=lambda pair: pair[0], reverse=True) # highest relevance first
    return reranked[:top_n]

#------------- FRONT END (html/css) -------------

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ARS Made Easy</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Source+Sans+3:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --ink: #1c1914;
            --muted: #5c564c;
            --paper: #ffefe0;
            --card: #fffaf4;
            --line: #e0d4c2;
            --copper: #b45309;
            --copper-dark: #9a3412;
            --canyon: #7c2d12;
            --sky: #1e4d5a;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            font-family: "Source Sans 3", "Segoe UI", sans-serif;
            color: var(--ink);
            background: var(--paper);
        }

        .wrap {
            max-width: 760px;
            width: 100%;
            margin: 0 auto;
            padding: 48px 22px 28vh;
            flex: 1;
            position: relative;
            z-index: 1;
        }

        .landscape {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 0;
            width: 100%;
            height: min(48vh, 440px);
            pointer-events: none;
            background:
                linear-gradient(
                    to bottom,
                    var(--paper) 0%,
                    rgba(255, 239, 224, 0.92) 14%,
                    rgba(255, 225, 189, 0.55) 34%,
                    rgba(255, 175, 78, 0.18) 58%,
                    transparent 100%
                ),
                url("/static/desert.svg") center bottom / cover no-repeat;
        }

        h1 {
            margin: 0;
            font-family: Fraunces, Georgia, serif;
            font-size: clamp(1.7rem, 4vw, 2.35rem);
            font-weight: 650;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }

        .lede {
            margin: 10px 0 28px;
            color: var(--muted);
            max-width: 42em;
            line-height: 1.5;
        }

        form {
            display: flex;
            gap: 8px;
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 8px;
            box-shadow: 0 12px 32px rgba(28, 25, 20, 0.12);
        }

        input[name="q"] {
            flex: 1;
            min-width: 0;
            border: 0;
            background: transparent;
            font: inherit;
            font-size: 1.05rem;
            padding: 12px 14px;
            color: var(--ink);
            outline: none;
        }

        button {
            border: 0;
            border-radius: 10px;
            background: var(--copper);
            color: #fff7ed;
            font: inherit;
            font-weight: 600;
            padding: 12px 18px;
            cursor: pointer;
        }

        button:hover { background: var(--copper-dark); }
        select[name="title"],
        select[name="chapter"] {
            flex: 0 0 auto;
            max-width: 140px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            border: 0;
            background: transparent;
            font: inherit;
            font-size: 0.95rem;
            color: var(--muted);
            padding: 12px 8px;
            outline: none;
            cursor: pointer;
        }

        .meta {
            margin: 22px 0 8px;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .result {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 18px 20px 16px;
            margin-top: 12px;
            box-shadow: 0 8px 20px rgba(28, 25, 20, 0.05);
        }

        .cite {
            font-family: Fraunces, Georgia, serif;
            font-size: 1.12rem;
            font-weight: 650;
            line-height: 1.35;
            margin: 0 0 8px;
        }

        .section-num {
            color: var(--canyon);
        }

        .blurb {
            margin: 0 0 12px;
            color: #3f3a33;
            line-height: 1.55;
        }

        .blurb em {
            font-style: italic;
            color: #3f3a33;
        }

        .source {
            color: var(--sky);
            font-size: 0.88rem;
            text-decoration: none;
            word-break: break-all;
        }

        mark {
            background: rgba(180, 83, 9, 0.22);
            color: inherit;
            padding: 0 2px;
            border-radius: 3px;
        }

        .source:hover { text-decoration: underline; }

        .empty {
            margin-top: 36px;
            padding: 28px 24px;
            border: 1px dashed var(--line);
            border-radius: 14px;
            color: var(--muted);
            background: var(--card);
        }

        .empty strong { color: var(--ink); }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Arizona Revised Statutes</h1>
        <p class="lede">Search the Arizona Revised Statutes (Arizona Law) by keyword. Headings rank higher than body text, and rarer legal terms carry more weight.</p>
        <form>
            <input type="text" name="q" value="__QUERY__" placeholder="Try assault, theft, sentencing, etc..." aria-label="Search statutes" autofocus>
            <select name="title" aria-label="Filter by title" onchange="this.form.submit()">
                <option value="">All titles</option>
                __TITLE_OPTIONS__
            </select>
            <select name="chapter" aria-label="Filter by chapter">
                <option value="">All chapters</option>
                __CHAPTER_OPTIONS__
            </select>
            <button type="submit">Search</button>
        </form>
        __BODY__
    </div>
    <div class="landscape" aria-hidden="true"></div>
</body>
</html>
"""

#------------- BACK END (Flask routes) -------------

@app.route("/")
def home():
    query = request.args.get("q", "")
    selected_title = request.args.get("title", "")
    selected_chapter = request.args.get("chapter", "")

    # Build title dropdown options
    title_options = ""
    for t in get_all_titles(statutes):
        display = f"Title {t} — {TITLE_NAMES.get(t, 'Unknown')}"
        selected = "selected" if t == selected_title else ""
        title_options += f'<option value="{escape(t)}" {selected}>{escape(display)}</option>'

    # Build chapter dropdown, scoped to the selected title (empty if no title chosen)
        # Note: learn how to custom-design dropdown menus to match website's design using CSS or JavaScript, default IOS dropdowns are ugly and difficult to read
    chapter_options = ""
    if selected_title:
        for ch in get_chapters_for_title(statutes, selected_title):
            display_name = ch.split("—")[-1].strip() if "—" in ch else ch
            selected = "selected" if ch == selected_chapter else ""
            chapter_options += f'<option value="{escape(ch)}" {selected}>{escape(display_name)}</option>'

    if not query:
        body = """
        <div class="empty">
            <strong>Start with a term, or pick a title to browse.</strong>
            Results open the official statute on azleg.gov.
        </div>
        """
        #rewrite all of the text above to sound less stupid 
        page = PAGE_TEMPLATE.replace("__QUERY__", str(escape(query))).replace("__BODY__", body)
        page = page.replace("__TITLE_OPTIONS__", title_options).replace("__CHAPTER_OPTIONS__", chapter_options)
        return page

    query_words = query.lower().split()
    candidates, best_snippets = hybrid_search(query, top_n=200, 
                               title=selected_title or None, 
                               chapter=selected_chapter or None)

    results = []
    if candidates:
        results = rerank(query, candidates, best_snippets, top_n=25)
        results = [(score, r) for score, r in results if score >= RERANK_MIN_SCORE]
    if not results or results[0][0] < RERANK_NO_MATCH:
        body = f"""
        <div class="empty">
            <strong>No strong matches for "{escape(query)}".</strong>
            Try a different term, a more specific phrase, or check the spelling.
        </div>
        """
    else:
        count = len(results) # number of results shown, not 200 candidates
        label = "result" if count == 1 else "results"
        body = f'<p class="meta">{count} {label} for "{escape(query)}"</p>'

        for score, r in results:
            if r.get("summary"):
                description = f"<em>{highlight(r['summary'], query_words)}</em>"
            else:
                snippet = get_snippet(r, query_words)
                description = f"...{highlight(snippet, query_words)}..."

            heading_html = highlight(r["heading"], query_words)

            body += f"""
            <article class="result">
                <h2 class="cite"><span class="section-num">§{escape(r['section'])}</span> — {heading_html}</h2>
                <p class="blurb">{description}</p>
                <a class="source" href="{escape(r['url'])}" target="_blank" rel="noopener noreferrer">{escape(r['url'])}</a>
            </article>
            """

    page = PAGE_TEMPLATE.replace("__QUERY__", str(escape(query))).replace("__BODY__", body)
    page = page.replace("__TITLE_OPTIONS__", title_options).replace("__CHAPTER_OPTIONS__", chapter_options)
    return page

# use to run locally
if __name__ == "__main__":
    app.run(debug=True, port=5001)

#use to run on google cloudrun
#if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 8080))
    #app.run(host="0.0.0.0", port=port)

# TO DO:
    # finish scraping titles, chapters
    # improve search ranking and relevance
    # add more comprehensive filtering options
    # enhance UI for better user experience
    # implement caching for frequently accessed queries
    # finish summarizing statutes, lowkey maybe restart and use a different model that doesn't immediately run out of credits
    # maybe add a dropdown under each statute that has additional information past the ai summary, such as:
        # related statutes
        # historical context
        # who the statute applies to
        # what conduct the statute addresses
        # penalties/legal consequences 
        # exceptions or defenses under the statute
        # any other information that may be important or relevant. 
    # add pages at the bottom of the search results. Right now, there's a limit on how many results can be on one page, but there's no way to access the other pages. 