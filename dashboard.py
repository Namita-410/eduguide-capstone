"""
EduGuide — Streamlit deployment app
=====================================
Loads the artifacts produced by EduGuide.ipynb (Sections 7-8):
  - finetuned_eduguide_adapter/   (LoRA adapter + tokenizer)
  - artifacts/faiss_index/eduguide.index       (FAISS retrieval index)
  - artifacts/faiss_index/chunk_lookup.csv     (chunk text + source metadata)

Run with:  streamlit run dashboard.py
Run this from the same directory where those three paths exist (i.e. wherever
you downloaded them from Colab after Section 8's save step).
"""

import os

# This app only uses PyTorch, never TensorFlow -- but transformers and
# sentence-transformers both probe for a TF backend at import time, which
# fails hard if TensorFlow + Keras 3 happen to be installed (a version
# combo transformers doesn't support). Setting USE_TF=0 before any
# transformers-related import skips that check entirely rather than
# requiring an extra tf-keras install just to satisfy an unused code path.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st
import torch

torch.set_num_threads(os.cpu_count())  # use all available CPU cores for inference
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Page config + design system
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EduGuide", page_icon="📖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --paper: #FAF8F3;
    --paper-raised: #FFFDF8;
    --ink: #1F2933;
    --ink-soft: #6B6459;
    --accent: #2F5233;
    --accent-soft: #E7EEE6;
    --gold: #A5730A;
    --rule: #E4DFD3;
}

.stApp { background-color: var(--paper); }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

h1, h2, h3 { font-family: 'Lora', serif; color: var(--ink); font-weight: 600; }

.eg-header {
    border-bottom: 1px solid var(--rule);
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}
.eg-title { font-family: 'Lora', serif; font-weight: 700; font-size: 2.1rem; margin: 0; color: var(--ink); }
.eg-subtitle { font-family: 'Inter', sans-serif; color: var(--ink-soft); font-size: 1rem; margin-top: 0.35rem; }

.eg-example-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.4rem; }

.eg-answer-card {
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1.3rem 1.5rem;
    margin-top: 1.2rem;
    line-height: 1.65;
}

.eg-sources {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px dashed var(--rule);
    font-size: 0.88rem;
    color: var(--ink-soft);
}
.eg-sources a { color: var(--gold); text-decoration: none; }
.eg-sources a:hover { text-decoration: underline; }

.stButton>button {
    background-color: var(--accent);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1.3rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}
.stButton>button:hover { background-color: #24401f; color: white; }

.eg-example-row .stButton>button {
    background-color: var(--accent-soft);
    color: var(--accent);
    font-size: 0.85rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
}
.eg-example-row .stButton>button:hover { background-color: #d8e3d6; color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model + retrieval artifacts (cached so this only runs once per session)
# ---------------------------------------------------------------------------
ADAPTER_PATH = "finetuned_eduguide_adapter"
FAISS_INDEX_PATH = "artifacts/faiss_index/eduguide.index"
CHUNK_LOOKUP_PATH = "artifacts/faiss_index/chunk_lookup.csv"


@st.cache_resource(show_spinner="Loading EduGuide's model and knowledge base...")
def load_artifacts():
    missing = [p for p in [ADAPTER_PATH, FAISS_INDEX_PATH, CHUNK_LOOKUP_PATH] if not os.path.exists(p)]
    if missing:
        return None

    # Load the tokenizer fresh from the base checkpoint rather than from
    # ADAPTER_PATH. LoRA fine-tuning never modifies the tokenizer/vocabulary,
    # so this is functionally identical -- and it avoids a real failure mode
    # where a tokenizer saved by one transformers version (e.g. in Colab)
    # can't be read by a different transformers version installed locally.
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(FAISS_INDEX_PATH)
    chunks_df = pd.read_csv(CHUNK_LOOKUP_PATH)

    return {"tokenizer": tokenizer, "model": model, "embedder": embedder,
            "index": index, "chunks_df": chunks_df}


def retrieve_context(artifacts, query, k=3):
    q_emb = artifacts["embedder"].encode([query]).astype("float32")
    _, indices = artifacts["index"].search(q_emb, k)
    return artifacts["chunks_df"].iloc[indices[0]]


def answer_question(artifacts, question, topic, difficulty, max_new_tokens=70):
    retrieved = retrieve_context(artifacts, question, k=2)  # fewer chunks -> shorter prompt, faster
    context_block = " ".join(retrieved["chunk_text"].tolist())
    prompt = (f"[Topic: {topic}] [Difficulty: {difficulty}] "
              f"Answer the question using only the context below.\n"
              f"Context: {context_block}\nQuestion: {question}")

    tokenizer, model = artifacts["tokenizer"], artifacts["model"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            num_beams=1,  # plain greedy -- no beam search overhead. repetition_penalty and
                          # no_repeat_ngram_size still handle most of the anti-repetition work.
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    sources = retrieved[["source_url"]].drop_duplicates()["source_url"].tolist()
    return answer, sources, retrieved


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="eg-header">
    <p class="eg-title">EduGuide</p>
    <p class="eg-subtitle">An AI learning assistant that answers from real documentation, adjusted to your level.</p>
</div>
""", unsafe_allow_html=True)

artifacts = load_artifacts()

if artifacts is None:
    st.warning(
        "EduGuide's trained model and knowledge base weren't found in this folder. "
        "Run Sections 7-8 of EduGuide.ipynb first, then place the resulting "
        f"`{ADAPTER_PATH}/` folder and `artifacts/` folder next to this file."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
TOPICS = ["All topics", "Python", "Machine Learning", "NLP", "Deep Learning", "Data Science", "General Programming"]

col1, col2 = st.columns([2, 1])
with col1:
    topic_choice = st.selectbox("Topic", TOPICS, index=0)
    # "All topics" is a UI-only label distinct from the real "General Programming"
    # topic; map it to the same "General" fallback used during training/testing.
    topic = "General" if topic_choice == "All topics" else topic_choice
with col2:
    difficulty = st.radio("Explain at", ["Beginner", "Intermediate"], horizontal=True)

# ---------------------------------------------------------------------------
# Example questions (from the plan's live-demo set)
# ---------------------------------------------------------------------------
EXAMPLES = [
    "What is supervised learning?",
    "What is the difference between precision and recall?",
    "Show me Python code for reading a CSV file using pandas.",
    "Why can a model have high accuracy but still perform poorly?",
    "Explain overfitting to a beginner with an example.",
]

if "question_text" not in st.session_state:
    st.session_state.question_text = ""

st.markdown('<div class="eg-example-row">', unsafe_allow_html=True)
example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    with col:
        if st.button(example[:24] + ("…" if len(example) > 24 else ""), key=f"ex_{example}", help=example):
            st.session_state.question_text = example
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Question input + answer
# ---------------------------------------------------------------------------
question = st.text_input(
    "Ask your learning question",
    value=st.session_state.question_text,
    placeholder="e.g. What is a Python list comprehension?",
)

ask_clicked = st.button("Ask EduGuide", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Thinking..."):
        answer, sources, retrieved = answer_question(artifacts, question, topic, difficulty)

    st.markdown(f'<div class="eg-answer-card">{answer}</div>', unsafe_allow_html=True)

    if sources:
        source_links = " &nbsp;·&nbsp; ".join(f'<a href="{s}" target="_blank">{s.split("/")[2]}</a>' for s in sources)
        st.markdown(f'<div class="eg-sources">Sources: {source_links}</div>', unsafe_allow_html=True)

    with st.expander("Show retrieved context"):
        for _, row in retrieved.iterrows():
            st.markdown(f"**{row.get('heading', row.get('topic', 'Context'))}**")
            st.write(row["chunk_text"])
            st.caption(row["source_url"])
            st.divider()

elif ask_clicked:
    st.info("Type a question first.")
