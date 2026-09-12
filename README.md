# EduGuide — AI Learning Assistant for Education & Training

An AI-powered academic learning assistant that answers technical questions across Python,
Machine Learning, NLP, Deep Learning, Data Science, and General Programming, adapted to
the learner's level (Beginner / Intermediate).

Built by combining **LoRA fine-tuning** of `google/flan-t5-base` with **retrieval-augmented
generation (RAG)** over a curated documentation corpus, evaluated as a three-way comparison
(base model / fine-tuned / fine-tuned + RAG) to isolate what each technique actually
contributes.

## Results

- **Dataset:** 270 QA pairs (26 hand-curated seed examples + 244 automatically generated
  from scraped documentation), 6 topics, near-balanced difficulty split (139 Beginner / 131
  Intermediate)
- **Training:** LoRA fine-tuning, 0.71% of parameters trainable, 25 epochs, final
  validation loss 2.964 (perplexity ≈19.4), converged around epoch 16-17 with no overfitting
- **Evaluation:** The fine-tuned + RAG model correctly grounded 3 of 5 test-query answers in
  verified source content, a clear improvement over the ungrounded zero-shot baseline (which
  hallucinated, e.g. confusing "overfitting" with body-fat percentage)

Full methodology, EDA, hypothesis testing, and a detailed limitations discussion (including
two data-pipeline bugs found and fixed during development) are in `EduGuide.ipynb`.

## Repository structure

```
EduGuide.ipynb              # full capstone notebook: data pipeline, EDA, training, evaluation
dashboard.py                 # Streamlit deployment app
build_seed_dataset.py        # generates the 26 hand-curated seed QA pairs
collect_docs.py              # scrapes documentation sources, builds the RAG chunk corpus,
                              # and generates additional QA pairs with difficulty inference
requirements.txt             # Python dependencies
```

## Running it

### 1. Regenerate the dataset
```bash
python build_seed_dataset.py
python collect_docs.py
```

### 2. Run the notebook
Open `EduGuide.ipynb` in Google Colab (a T4 GPU is recommended for the fine-tuning step)
and run top to bottom. This produces `finetuned_eduguide_adapter/` and
`artifacts/faiss_index/`.

### 3. Run the app
Place `finetuned_eduguide_adapter/` and `artifacts/` next to `dashboard.py`, then:
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Known limitations

- **Topic/difficulty imbalance:** Python skews 83% Beginner while Machine Learning skews
  87% Intermediate, traced to uneven availability of introductory-tutorial source pages
  per topic.
- **Retrieval coverage gaps:** questions about concepts absent from the source
  documentation (e.g. Python namespaces) can retrieve irrelevant context, and the model
  will sometimes fabricate a plausible-sounding but incorrect answer rather than abstaining.
- **CPU inference latency:** generation settings (beam width, max tokens, retrieved chunk
  count) were reduced from their quality-optimal values to get usable response times on
  CPU-only hardware; GPU inference or model quantization would close this gap.

## Dataset attribution

Documentation content scraped from official sources (Python, NumPy, pandas, scikit-learn,
Hugging Face Transformers, Sentence-Transformers, PyTorch, Git) under fair use for
educational/non-commercial purposes. See `collect_docs.py` for the full source list.
