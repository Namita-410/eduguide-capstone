"""
EduGuide - Documentation Collection & QA Generation
=====================================================
Run this in Colab (needs internet). It does two jobs:

  1. Knowledge corpus collection: scrapes official documentation pages into
     data/raw/docs_raw.csv -> this feeds the RAG/FAISS side of the pipeline.
  2. QA pair generation: turns cleaned doc sections into structured QA pairs
     appended to the seed dataset -> this feeds the LoRA fine-tuning side.

This mirrors the plan's two-stage data preparation:
    Educational documents -> Cleaning -> Dedup -> Chunking -> Embedding -> FAISS   (RAG)
    Educational content    -> Question generation -> Answer creation -> Quality
                               filtering -> Difficulty tagging -> Train/val split  (LoRA)

NOTE ON QA GENERATION: doing this well typically means prompting an LLM (e.g. via
the Anthropic or OpenAI API, or a local Hugging Face model) over each doc chunk to
produce a question/answer pair, then a human (you) spot-checking a sample for
quality before it goes into training. This script provides the scraping +
chunking + a lightweight heuristic-based generator as a v0 you can run without an
API key; swap in an LLM-based generator (see generate_qa_with_llm() stub below)
once you're ready, since it will produce noticeably better questions/answers.
"""

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# 1. SOURCES
# ---------------------------------------------------------------------------
# Keep this list short and high-signal rather than crawling entire doc sites --
# a handful of well-chosen conceptual/tutorial pages beats thousands of noisy
# API-reference pages for a *teaching* assistant (as opposed to a pure lookup tool).
SOURCES = [
    # (topic, url)
    ("Python", "https://docs.python.org/3/tutorial/datastructures.html"),
    ("Python", "https://docs.python.org/3/tutorial/controlflow.html"),
    ("Python", "https://docs.python.org/3/tutorial/errors.html"),
    ("Python", "https://numpy.org/doc/stable/user/absolute_beginners.html"),
    ("Python", "https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html"),
    ("Machine Learning", "https://scikit-learn.org/stable/modules/cross_validation.html"),
    ("Machine Learning", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
    ("Machine Learning", "https://scikit-learn.org/stable/modules/tree.html"),
    ("Machine Learning", "https://scikit-learn.org/stable/unsupervised_learning.html"),
    ("Machine Learning", "https://scikit-learn.org/stable/getting_started.html"),
    ("NLP", "https://huggingface.co/docs/transformers/tokenizer_summary"),
    ("NLP", "https://scikit-learn.org/stable/modules/feature_extraction.html"),
    ("NLP", "https://huggingface.co/docs/transformers/quicktour"),
    # sbert.net removed: its page content includes unrendered markdown and decodes
    # with corrupted characters (e.g. permalink symbols becoming "ï"), and it only
    # ever yielded 2 low-value sections -- not worth the cleanup effort.
    ("Deep Learning", "https://pytorch.org/tutorials/beginner/basics/intro.html"),
    ("Deep Learning", "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"),
    ("Deep Learning", "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"),
    ("Data Science", "https://pandas.pydata.org/docs/user_guide/groupby.html"),
    ("Data Science", "https://pandas.pydata.org/docs/user_guide/merging.html"),
    ("Data Science", "https://pandas.pydata.org/docs/user_guide/10min.html"),
    ("General Programming", "https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control"),
    ("General Programming", "https://docs.python.org/3/tutorial/modules.html"),
]

HEADERS = {"User-Agent": "EduGuide-Capstone-Bot/1.0 (educational project)"}


def fetch_page(url, retries=2, delay=1.5):
    """Fetch a page's HTML with basic retry/backoff. Returns None on failure
    rather than raising, so one bad URL doesn't kill the whole collection run."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            # requests sometimes mis-guesses encoding (e.g. falls back to
            # ISO-8859-1 when a server omits charset), which corrupts symbols
            # like the paragraph/permalink mark into garbage characters.
            # Modern doc sites are overwhelmingly UTF-8, so force it explicitly.
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            if attempt == retries:
                print(f"  [warn] failed to fetch {url}: {e}")
                return None
            time.sleep(delay)


def heading_depth(raw_heading):
    """How deeply nested a heading's outline number is (e.g. '8.2.3.10.' -> 4,
    '3.1' -> 2, unnumbered -> 0). Deeply-nested sections in Sphinx-style docs
    are reliably narrower/more advanced sub-topics (e.g. individual function
    reference entries) than top-level or unnumbered conceptual sections --
    this is used as one signal for difficulty inference below."""
    m = re.match(r"^(\d+(\.\d+)*)\.?\s*", raw_heading)
    if not m:
        return 0
    return m.group(1).count(".") + 1


def extract_text_sections(html, min_words=40):
    """Pull out heading + following-paragraph 'sections' from a doc page.
    Returns a list of dicts: {heading, text, heading_depth}. This is a simple
    heuristic extractor -- documentation sites vary a lot in structure, so
    always spot-check a few outputs per source rather than trusting it blindly."""
    soup = BeautifulSoup(html, "html.parser")

    # Strip navigation/boilerplate that would otherwise pollute the corpus
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    sections = []
    headings = soup.find_all(["h1", "h2", "h3"])
    for h in headings:
        raw_heading = h.get_text(strip=True)
        depth = heading_depth(raw_heading)
        heading_text = clean_heading(raw_heading)
        if not heading_text:
            continue
        # Collect sibling paragraph/code text until the next heading
        parts = []
        for sib in h.find_next_siblings():
            if sib.name in ("h1", "h2", "h3"):
                break
            if sib.name in ("p", "li", "pre", "code"):
                t = sib.get_text(" ", strip=True)
                if t:
                    parts.append(t)
        body = " ".join(parts)
        word_count = len(body.split())
        if word_count >= min_words:
            sections.append({"heading": heading_text, "text": body, "heading_depth": depth})
    return sections


def clean_text(text):
    """Basic text cleaning: collapse whitespace, strip stray unicode artifacts
    that documentation sites commonly leave in (e.g. permalink '¶' anchors)."""
    text = text.replace("¶", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_heading(heading):
    """Sphinx/MkDocs-style doc sites number their headings (e.g. '8.2.3.10.
    Customizing the vectorizer classes') and append a permalink anchor
    character (e.g. a trailing '#' or '¶') that H tags carry as text. Left
    in, these leak straight into auto-generated questions as visible noise
    -- e.g. 'What is 8.2.3.10.Customizing the vectorizer classes#?' -- which
    is exactly the kind of broken-looking output this cleaning step exists
    to prevent before it reaches the QA dataset."""
    heading = heading.replace("¶", "").rstrip("#").strip()
    # Strip a leading numeric outline prefix like "8.2.3.10." or "3.1 "
    heading = re.sub(r"^\d+(\.\d+)*\.?\s*", "", heading)
    return heading.strip()


def collect_raw_docs():
    """Job 1: scrape all SOURCES -> data/raw/docs_raw.csv"""
    rows = []
    for topic, url in SOURCES:
        print(f"Fetching [{topic}] {url}")
        html = fetch_page(url)
        if html is None:
            continue
        sections = extract_text_sections(html)
        for sec in sections:
            rows.append({
                "topic": topic,
                "source_url": url,
                "heading": sec["heading"],
                "text": clean_text(sec["text"]),
                "heading_depth": sec["heading_depth"],
            })
        time.sleep(1)  # be polite to doc servers

    df = pd.DataFrame(rows)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/docs_raw.csv", index=False)
    print(f"\nCollected {len(df)} sections across {df['source_url'].nunique() if len(df) else 0} pages")
    return df


# ---------------------------------------------------------------------------
# 2. CHUNKING (for the RAG/FAISS side)
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=400, overlap=80):
    """Word-based sliding-window chunking with overlap, so a fact split across
    a chunk boundary still has a reasonable chance of appearing whole in at
    least one chunk -- important for retrieval quality."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= 20:  # drop tiny trailing scraps
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_doc_chunks(docs_raw_df):
    """Job 1b: docs_raw.csv -> data/processed/doc_chunks.csv"""
    rows = []
    for _, r in docs_raw_df.iterrows():
        for i, chunk in enumerate(chunk_text(r["text"])):
            rows.append({
                "chunk_id": f"{r['source_url']}#{i}",
                "topic": r["topic"],
                "heading": r["heading"],
                "source_url": r["source_url"],
                "chunk_text": chunk,
            })
    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/doc_chunks.csv", index=False)
    print(f"Built {len(df)} chunks")
    return df


# ---------------------------------------------------------------------------
# 3. QA PAIR GENERATION (for the LoRA fine-tuning side)
# ---------------------------------------------------------------------------
# Pages that are explicitly titled/framed as introductory tutorials rather
# than reference material -- a real signal for difficulty, not a guess.
BEGINNER_SOURCE_HINTS = {
    "https://numpy.org/doc/stable/user/absolute_beginners.html",
    "https://pytorch.org/tutorials/beginner/basics/intro.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html",
    "https://scikit-learn.org/stable/getting_started.html",
    "https://huggingface.co/docs/transformers/quicktour",
    "https://pandas.pydata.org/docs/user_guide/10min.html",
}


def infer_difficulty(source_url, heading_depth, word_count):
    """v0.3 heuristic difficulty inference.
      1. A source in BEGINNER_SOURCE_HINTS is trusted as Beginner for its
         shallow-to-moderately-nested sections (depth <= 2) -- covering the
         vast majority of a tutorial page's real content -- but NOT
         unconditionally: a deeply-nested, long subsection even within an
         intro-framed page (e.g. 'Nested List Comprehensions' several
         levels into a Python tutorial) can still be denser than the page's
         overall framing suggests, so it falls through to a length check
         instead of being auto-labeled Beginner.
      2. Outside a hinted source, a shallow/unnumbered AND short section is
         still treated as Beginner -- the only path available to topics
         with no hinted source at all.
    v0.1 made Beginner structurally impossible for unhinted topics. v0.2
    fixed that but over-corrected the other way for heavily-hinted topics
    (e.g. Python jumped from a balanced split to 73 Beginner vs 3
    Intermediate) by trusting every hinted-source section unconditionally.
    This version narrows that back down. Still a heuristic, not a trained
    classifier -- swap in generate_qa_with_llm()'s DIFFICULTY output for a
    more reliable label before a final training run if you have LLM API
    access."""
    if source_url in BEGINNER_SOURCE_HINTS:
        if heading_depth <= 2:
            return "Beginner"
        return "Beginner" if word_count < 80 else "Intermediate"
    if heading_depth <= 1 and word_count < 80:
        return "Beginner"
    return "Intermediate"


def smart_truncate(text, max_words=140, min_words=50):
    """Truncate long text to at most max_words, but back off to the nearest
    preceding sentence boundary so answers end on a complete thought instead
    of being cut off mid-sentence (the old hard 60-word cutoff's main flaw,
    e.g. '...will tell you the total number of elements of t'). Falls back
    to a hard cut only if no usable sentence boundary exists in range."""
    words = text.split()
    if len(words) <= max_words:
        return text
    window = " ".join(words[:max_words])
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut == -1:
        return " ".join(words[:max_words]) + "."
    candidate = window[:cut + 1]
    if len(candidate.split()) < min_words:
        # sentence boundary came too early -- a tiny answer is worse than a
        # slightly-abrupt-but-substantial one, so keep the fuller window
        return window + "."
    return candidate


def generate_qa_heuristic(docs_raw_df):
    """v0 generator: turns each doc section's heading into a 'What is/How does
    <heading>...' question with the section text as context and a
    sentence-boundary-truncated answer, with difficulty inferred per-row
    (see infer_difficulty) rather than a single hardcoded default. This is
    still a weak baseline compared to an LLM-based generator -- questions are
    mechanically templated from headings -- but it needs no API key and gets
    the pipeline end-to-end runnable immediately. Replace with
    generate_qa_with_llm() before your final training run for real quality."""
    rows = []
    for i, r in docs_raw_df.iterrows():
        heading = r["heading"]
        text = r["text"]
        if len(text.split()) < 30:
            continue
        question = f"What is {heading}?" if not heading.lower().startswith(("what", "how", "why")) else heading
        answer = smart_truncate(text)
        difficulty = infer_difficulty(r["source_url"], r["heading_depth"], len(text.split()))
        rows.append({
            "id": f"auto_{i:04d}",
            "topic": r["topic"],
            "subtopic": heading,
            "question": question,
            "context": text,
            "answer": answer,
            "difficulty": difficulty,
            "source": r["source_url"].split("/")[2],
            "source_url": r["source_url"],
        })
    return pd.DataFrame(rows)


def generate_qa_with_llm(docs_raw_df, call_llm_fn):
    """v1 generator (recommended before final training). Pass in a function
    call_llm_fn(prompt: str) -> str that hits whatever LLM API you have access
    to (Anthropic API, OpenAI, or a local Hugging Face model), and this will
    prompt it per section to produce a genuinely well-formed question + answer
    + difficulty label, then parse the structured response.

    Example call_llm_fn using the Anthropic API:

        import anthropic
        client = anthropic.Anthropic(api_key="...")
        def call_llm_fn(prompt):
            msg = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text
    """
    rows = []
    prompt_template = """Given this documentation excerpt, write ONE clear student-facing
question and a concise, accurate answer grounded only in the excerpt. Also label the
question's difficulty as Beginner or Intermediate. Respond in exactly this format:
QUESTION: <question>
ANSWER: <answer>
DIFFICULTY: <Beginner or Intermediate>

Excerpt (topic: {topic}, section: {heading}):
{text}"""

    for i, r in docs_raw_df.iterrows():
        if len(r["text"].split()) < 30:
            continue
        prompt = prompt_template.format(topic=r["topic"], heading=r["heading"], text=r["text"][:1200])
        raw = call_llm_fn(prompt)
        q_match = re.search(r"QUESTION:\s*(.+)", raw)
        a_match = re.search(r"ANSWER:\s*(.+?)(?=DIFFICULTY:|$)", raw, re.DOTALL)
        d_match = re.search(r"DIFFICULTY:\s*(Beginner|Intermediate)", raw)
        if not (q_match and a_match):
            continue
        rows.append({
            "id": f"llm_{i:04d}",
            "topic": r["topic"],
            "subtopic": r["heading"],
            "question": q_match.group(1).strip(),
            "context": r["text"],
            "answer": a_match.group(1).strip(),
            "difficulty": d_match.group(1) if d_match else "Intermediate",
            "source": r["source_url"].split("/")[2],
            "source_url": r["source_url"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. QUALITY FILTERING
# ---------------------------------------------------------------------------
def quality_filter(qa_df, min_answer_words=8, max_answer_words=200):
    """Drop rows that are too short to be useful, too long to be a clean
    answer, exact-duplicate questions, or missing required fields. Document
    what's dropped and why -- this is graded as part of preprocessing rigor."""
    before = len(qa_df)
    qa_df = qa_df.dropna(subset=["question", "answer"]).copy()
    qa_df = qa_df.drop_duplicates(subset=["question"])
    qa_df["answer_word_count"] = qa_df["answer"].str.split().str.len()
    qa_df = qa_df[
        (qa_df["answer_word_count"] >= min_answer_words)
        & (qa_df["answer_word_count"] <= max_answer_words)
    ]
    qa_df = qa_df.drop(columns=["answer_word_count"])
    print(f"Quality filter: {before} -> {len(qa_df)} rows "
          f"({before - len(qa_df)} dropped: dupes/too short/too long)")
    return qa_df.reset_index(drop=True)


if __name__ == "__main__":
    print("=== Phase 2 Step 1: Scraping documentation ===")
    docs_raw = collect_raw_docs()

    print("\n=== Phase 2 Step 2: Chunking for RAG ===")
    doc_chunks = build_doc_chunks(docs_raw)

    print("\n=== Phase 2 Step 3: Generating QA pairs (heuristic v0) ===")
    qa_auto = generate_qa_heuristic(docs_raw)
    qa_auto = quality_filter(qa_auto)

    print("\n=== Phase 2 Step 4: Merging with hand-curated seed set ===")
    seed = pd.read_csv("data/raw/education_dataset_seed.csv")
    combined = pd.concat([seed, qa_auto], ignore_index=True)
    combined = combined.drop_duplicates(subset=["question"]).reset_index(drop=True)
    os.makedirs("data/processed", exist_ok=True)
    combined.to_csv("data/processed/education_dataset.csv", index=False)
    print(f"\nFinal combined dataset: {len(combined)} QA pairs")
    print(combined["topic"].value_counts())
