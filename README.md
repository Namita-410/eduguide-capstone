# EduGuide — AI Learning Assistant for Education & Training

> **Industry LLM Bot Capstone** · Advanced Certification in Data Science & AI  
> IIT Guwahati × AlmaBetter · Individual Project · Namita Sharma

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Namita-410/eduguide-capstone/blob/main/EduGuide_.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://travel-mlops-capstone-frw8erw85t38rcf562fr6s.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗-FLAN--T5--base-yellow)](https://huggingface.co/google/flan-t5-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is EduGuide?

EduGuide is a domain-specific conversational AI assistant built for the **Education and Training** industry. It helps students understand technical concepts across Python, Machine Learning, NLP, Deep Learning, Data Science, and General Programming — with answers that are:

- **Grounded** in official documentation (not hallucinated from parametric memory)
- **Difficulty-aware** — the same question gets a different depth of explanation for a Beginner vs an Intermediate learner
- **Source-cited** — every answer includes the documentation page it was retrieved from

The system combines two complementary techniques applied to the same curated educational corpus: **LoRA fine-tuning** of Google's FLAN-T5-base for domain-adapted response style, and **FAISS-based Retrieval-Augmented Generation (RAG)** for grounded, verifiable answers at inference time.

---

## Live Demo

🌐 **Streamlit App:** https://travel-mlops-capstone-frw8erw85t38rcf562fr6s.streamlit.app/

---

## Project Architecture

```
Student query
     │
     ▼
┌─────────────────────────────────┐
│   Prompt construction           │
│   Topic + Difficulty prefix     │
│   "Topic: ML. Difficulty:       │
│    Beginner. Question: ..."     │
└────────────┬────────────────────┘
             │
     ┌───────▼────────┐
     │  FAISS Index   │  ← 288 chunks from 21 doc pages
     │  (RAG Layer)   │     all-MiniLM-L6-v2 embeddings
     │  k=3 retrieval │     distance guardrail @ 0.75
     └───────┬────────┘
             │  retrieved context
     ┌───────▼────────────────────┐
     │  LoRA Fine-Tuned FLAN-T5   │  ← google/flan-t5-base
     │  (Model B + RAG = Model C) │     r=16, α=32, q+v layers
     │  1,769,472 trainable params│     0.71% of 249M total
     └───────┬────────────────────┘
             │
     ┌───────▼────────────────────┐
     │  Generated answer          │
     │  + source citation         │
     └────────────────────────────┘
```

---

## Three-Model Comparison

| | Model A | Model B | Model C ✅ |
|---|---|---|---|
| **Description** | Base FLAN-T5, zero-shot | LoRA fine-tuned FLAN-T5 | Fine-tuned + FAISS RAG |
| **Fine-tuning** | ❌ None | ✅ LoRA (r=16) | ✅ LoRA (r=16) |
| **Retrieval** | ❌ None | ❌ None | ✅ FAISS k=3 |
| **Difficulty-aware** | ❌ Ignores prefix | ✅ Respects prefix | ✅ Respects prefix |
| **Source-cited** | ❌ | ❌ | ✅ |
| **Hallucination risk** | High | Medium | Low (guardrail) |
| **5-query test** | 0/5 accurate | Fluent but ungrounded | 2/5 accurate + sourced, 1/5 correctly declined |
| **Chosen as final** | ❌ | ❌ | ✅ |

Model C was chosen as the final model — the only one that was both correct and independently verifiable on the test queries, with a retrieval-distance guardrail that causes it to decline rather than guess when no relevant context is found.

---

## Dataset

| Property | Value |
|---|---|
| Total QA pairs | 270 |
| Training split | 229 (85%) |
| Validation split | 41 (15%) |
| Difficulty split | 139 Beginner / 131 Intermediate (~51/49%) |
| Topics | Python, Machine Learning, NLP, Deep Learning, Data Science, General Programming |
| Data origin | 25 hand-curated seed pairs + 245 auto-generated from scraped docs |
| Source diversity | scikit-learn.org, pandas.pydata.org, docs.python.org, numpy.org, pytorch.org, huggingface.co |
| Schema | id, topic, subtopic, question, context, answer, difficulty, source, source_url |

### Data Pipeline

```
21 official doc pages
        │
        ▼
BeautifulSoup scraping
(headings + body sections)
        │
        ▼
Text cleaning
(permalink anchors, numbering, whitespace, unicode NFKD)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
400-word sliding-window chunks            Heuristic QA generation
(80-word overlap, for FAISS)              (heading → question,
→ data/processed/doc_chunks.csv            section → answer,
                                           difficulty inference)
                                                      │
                                                      ▼
                                           Quality filtering
                                           (8–200 word answers,
                                            dedup, null drop)
                                                      │
                                                      ▼
                                           Merge with seed dataset
                                           → data/processed/
                                             education_dataset.csv
```

---

## Model Details

### LoRA Fine-Tuning (Model B)

| Parameter | Value |
|---|---|
| Base model | `google/flan-t5-base` (249M parameters) |
| PEFT method | LoRA |
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Target modules | `q`, `v` (query and value attention projections) |
| Trainable parameters | 1,769,472 (0.71% of total) |
| Epochs | 25 |
| Learning rate | 1e-3 |
| Batch size | 8 |
| Training time | ~6 minutes (361.96s, 725 steps) on Colab T4 GPU |
| Train loss | 3.62 (epoch 1) → 3.04 (epoch 25) |
| Val loss | 3.17 → 2.96 (no overfitting observed) |

### FAISS RAG (Model C additions)

| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Embedding dimension | 384 |
| Index type | Flat L2 (exact search) |
| Total chunks | 288 (262 scraped + 26 seed contexts) |
| Retrieval k | 3 nearest chunks |
| Distance guardrail | 0.75 (calibrated from real in/out-of-corpus measurements) |
| Chunk size | 400 words |
| Chunk overlap | 80 words |

---

## Repository Structure

```
eduguide-capstone/
│
├── EduGuide_.ipynb                  # Main Colab notebook (all sections)
│
├── collect_docs.py                  # Doc scraping, chunking & QA generation
├── build_seed_dataset.py            # Hand-curated seed QA pairs builder
│
├── data/
│   ├── raw/
│   │   ├── education_dataset_seed.csv   # 25 hand-curated QA pairs
│   │   └── docs_raw.csv                 # Scraped doc sections
│   └── processed/
│       ├── education_dataset.csv        # Full 270-pair dataset
│       ├── doc_chunks.csv               # 288 FAISS chunks
│       ├── qa_train_split.jsonl         # 229 training examples
│       └── qa_val_split.jsonl           # 41 validation examples
│
├── artifacts/
│   ├── faiss_index/
│   │   ├── eduguide.index               # Saved FAISS index
│   │   └── chunk_lookup.csv             # Chunk ID → text mapping
│   └── eduguide_artifacts.zip           # All artifacts zipped
│
├── finetuned_eduguide_adapter/      # Saved LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
│
├── streamlit_app/
│   └── app.py                       # Streamlit dashboard
│
└── requirements.txt
```

---

## Quickstart

### Run the notebook in Colab

```
1. Open EduGuide_.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom
   (first run takes ~15 min to install deps + scrape docs + train)
```

### Run the Streamlit app locally

```bash
git clone https://github.com/Namita-410/eduguide-capstone.git
cd eduguide-capstone
pip install -r requirements.txt

# Make sure artifacts exist (run notebook first, or download from releases)
streamlit run streamlit_app/app.py
```

### Run in Colab with public URL

```python
!git clone https://github.com/Namita-410/eduguide-capstone.git
%cd eduguide-capstone
!pip install streamlit pyngrok --quiet

import subprocess, time
from pyngrok import ngrok

proc = subprocess.Popen(
    ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.headless=true"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(6)
print(ngrok.connect(8501))
```

---

## Requirements

```
transformers>=4.40.0
datasets>=2.19.0
accelerate>=0.30.0
peft>=0.10.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
torch>=2.2.0
evaluate>=0.4.0
langchain>=0.1.0
langchain-community>=0.0.32
streamlit>=1.33.0
beautifulsoup4>=4.12.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
scipy>=1.13.0
nltk>=3.8.0
contractions>=0.1.73
wordcloud>=1.9.0
```

---

## Key Results

- **Training loss:** 3.62 → 3.04 over 25 epochs, no overfitting
- **Difficulty balance:** 51% Beginner / 49% Intermediate (χ² test: p < 0.05, topic and difficulty are not independent)
- **Answer length hypothesis:** Intermediate answers are significantly longer than Beginner answers (Welch t-test: p < 0.001)
- **Context-answer correlation:** Pearson r = 0.65 between context length and answer length — longer source passages produce longer answers
- **Model C on 5-query test:** 2/5 accurate and sourced, 1/5 correctly declined via guardrail, 2/5 grounded-but-off-target (retrieved wrong chunk, not a hallucination)

---

## Limitations

- **Dataset size:** 270 pairs is small; sparse topic-difficulty cells (Python-Intermediate: 13 pairs, ML-Advanced: 0 pairs) limit coverage
- **Source bias:** ~85 pairs from scikit-learn.org — the model is stronger on sklearn queries than PyTorch
- **Evaluation:** No formal ROUGE/BLEU table exists; qualitative 5-query comparison is the primary evidence
- **Hallucination:** Flat L2 FAISS with a single distance guardrail does not eliminate all off-topic retrievals — 2/5 test queries retrieved a related but wrong chunk
- **FLAN-T5-base scale:** 249M parameters is modest; a larger model (T5-large, Mistral-7B) would likely show clearer fine-tuning gains

---

## Future Work

- Replace heuristic QA generator with LLM-based generation (stub already in `collect_docs.py` — `generate_qa_with_llm()`)
- Add re-ranking step to FAISS retrieval (cross-encoder scoring of top-k chunks before context injection)
- Implement LLM-as-judge evaluation framework for reliable quality scoring
- Expand to Advanced difficulty tier with expert-curated examples
- Deploy on Hugging Face Spaces for persistent public access
- Add multi-modal input (code screenshots, diagrams) for debugging assistance

---

## Acknowledgements

- **Base model:** [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) — Google Research
- **Embeddings:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — sentence-transformers
- **Documentation sources:** Python.org, NumPy, pandas, scikit-learn, Hugging Face, PyTorch, Git
- **Program:** Advanced Certification in Data Science & AI — IIT Guwahati × AlmaBetter

---

*EduGuide capstone — Namita Sharma · github.com/Namita-410/eduguide-capstone*
