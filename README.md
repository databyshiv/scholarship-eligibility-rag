# 🎓 Scholarship & Grant Eligibility Checker

A **Test-Driven Retrieval-Augmented Generation (RAG)** system that checks scholarship and grant eligibility by reading official policy documents and answering student queries with cited sources.

Built for the **RMIT WIL Project** on Test-Driven RAG.

---

## 📁 Project Structure

```
scholarship-rag/
├── data/
│   ├── raw/                          # Knowledge base documents
│   │   ├── kb1_rmit_scholarship_terms.txt
│   │   ├── kb2_rmit_rtp_policy.txt
│   │   └── kb3_arc_discovery_guidelines.txt
│   └── evaluation/
│       ├── gold_standard.csv         # 30 test cases with ground truth
│       ├── evaluation_results.csv    # Generated after evaluation run
│       └── evaluation_report.txt     # Generated after evaluation run
├── src/
│   ├── chunking.py                   # Semantic document chunking
│   ├── rag_pipeline.py               # Core RAG pipeline (ChromaDB + Ollama)
│   └── evaluation.py                 # Test-driven evaluation framework
├── demo/
│   └── app.py                        # Streamlit interactive web app
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start
### Step 1: Install Ollama (Local LLM)

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull qwen2.5:7b
```

> Other Ollama models also work: pass `--model llama3.1` to the evaluation, or pick one under
> **Advanced settings** in the app.

---

### Step 2: Install Python Dependencies

```bash
cd scholarship-rag
pip install -r requirements.txt
pip install -r requirements-ragas.txt   # only needed for the RAGAS evaluation
```

The embedding model (`BAAI/bge-small-en-v1.5`, about 130 MB) downloads automatically on first run.

---

### Step 3: Run the Evaluation (Test-Driven Baseline)

```bash
python src/evaluation.py        # 30 gold-standard cases -> accuracy report
python src/metrics.py           # precision, recall, F1, confusion matrix
python src/retrieval_eval.py    # dense vs hybrid retrieval (Hit@k, MRR), no LLM needed
python src/ragas_eval.py --limit 5   # RAGAS quality scores (slow; drop --limit for all 30)
```

**Output** (in `data/evaluation/`):
- `evaluation_results.csv`, `evaluation_report.txt`: per-case results and summary metrics
- `retrieval_report.txt`: retrieval comparison
- `ragas_report.txt`, `ragas_results.csv`: faithfulness, answer relevancy, context precision and recall

**Key Metrics Reported:**
- Overall Accuracy, Macro F1 and Weighted F1
- % Unanswered (Walert-style)
- Accuracy by Difficulty (SIMPLE / CONDITIONAL / EXCEPTION_TRAP)
- Fairness Analysis by Demographic (DOMESTIC / INTERNATIONAL / EQUITY)
- Exception Trap Precision

---

### Step 4: Run the Interactive Demo

```bash
streamlit run demo/app.py
```

Then open your browser to `http://localhost:8501`

The demo lets you:
- Ask natural language eligibility questions
- See the exact policy clauses retrieved
- View the LLM's reasoning and citation
- Get a clear verdict: ✅ ELIGIBLE / ❌ INELIGIBLE / ⚠️ CANNOT DETERMINE

---

## 📊 Evaluation Framework

This project implements a **robust, quantitative evaluation framework** as required by the assignment:

| Dimension | Metric | How It's Measured |
|-----------|--------|-------------------|
| **Effectiveness** | Overall Accuracy + % Unanswered | Correct predictions / total cases |
| **Conditional Logic Precision** | Exception Trap Accuracy | Accuracy on `EXCEPTION_TRAP` test cases specifically |
| **Faithfulness** | Source Attribution | Every response includes an exact quoted clause |
| **Fairness** | Demographic Parity | Accuracy compared across DOMESTIC / INTERNATIONAL / EQUITY groups |
| **Correctness of Sources** | Citation Match | Retrieved chunks contain the critical clause |

### Test Case Distribution (30 total)

| Source | Count | Focus |
|--------|-------|-------|
| KB-1 (RMIT Terms) | 10 | Part-time rules, prior scholarship disqualification, EFTSL limits |
| KB-2 (RMIT RTP Policy) | 10 | Concurrent income >75%, transfer rules, suspension triggers |
| KB-3 (ARC Guidelines) | 10 | DECRA two-strikes, PhD date cutoff, Future Fellowship three-strikes |

| Difficulty | Count | Purpose |
|------------|-------|---------|
| SIMPLE | 10 | Baseline retrieval accuracy |
| CONDITIONAL | 12 | Tests "if/then" reasoning |
| EXCEPTION_TRAP | 8 | Tests whether RAG catches disqualifying exception clauses |

---

## 🧠 How It Works

1. **Chunking** (`src/chunking.py`): Documents are split by semantic sections (headers) rather than fixed size. Exception clauses stay with their parent conditions.
2. **Embedding** (`src/rag_pipeline.py`): Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB.
3. **Retrieval**: User queries are embedded and the top-k most similar chunks are retrieved.
4. **Generation**: A local LLM (via Ollama) generates a verdict using ONLY the retrieved context, with strict instructions to cite exact clauses and respect exception conditions.
5. **Evaluation**: The 30 test cases measure accuracy across difficulty levels and demographics.

---

## 🔧 Customization

### Change the LLM Model
Edit `src/rag_pipeline.py`:
```python
rag = ScholarshipRAG(chunks, llm_model='mistral')  # or 'phi3', 'llama2', etc.
```

### Change Number of Retrieved Chunks
In the Streamlit app sidebar, use the slider (default: 5).
In code:
```python
rag.answer(query, n_results=7)
```

### Add More Test Cases
Edit `data/evaluation/gold_standard.csv` and re-run `python src/evaluation.py`.

---

## ⚠️ Ethical Considerations

- **Not Official Advice:** This is a prototype. Users must verify with the School of Graduate Research.
- **Bias Awareness:** The fairness metric checks if accuracy differs across demographic groups.
- **Transparency:** Every answer shows the exact source clause used.
- **Privacy:** No student data is stored or transmitted to external APIs (all local).

---

## 📚 Data Sources

| Document | Source |
|----------|--------|
| KB-1 | RMIT Research Scholarship Terms & Conditions (rmit.edu.au) |
| KB-2 | RMIT HDR and RTP Scholarships Policy (policies.rmit.edu.au) |
| KB-3 | ARC Discovery Program Grant Guidelines — Fellowships 2024 Variation 1 (arc.gov.au) |

All documents are publicly available and used for educational purposes.

---

## 👥 Team & Methodology

This project follows **lean/agile methodology**:
- Sprint-based development
- Git version control with clear commit history
- Test-driven approach: evaluation dataset built BEFORE pipeline optimization
- Iterative improvement based on quantitative metrics

---

**License:** Educational Use Only  
**Course:** RMIT WIL Project — Test-Driven RAG

---

## 🔎 Retrieval (update)

Retrieval is now **dense (embedding) search** by default, using `BAAI/bge-small-en-v1.5`
stored in ChromaDB. The question and (question + student profile) are each embedded and
the two rankings are merged. The older **hybrid** mode (dense + BM25 keyword search) is
kept for comparison:

```bash
python src/evaluation.py                       # dense (default)
python src/evaluation.py --retrieval hybrid    # dense + BM25
python src/retrieval_eval.py                   # compare both, no LLM needed (Hit@k, MRR)
```

## 📏 RAGAS evaluation

RAGAS scores the quality of retrieval and generation with a local judge LLM (Ollama):
faithfulness, answer relevancy, context precision and context recall (all 0-1).

```bash
pip install -r requirements-ragas.txt
python src/evaluation.py            # saves answers + retrieved chunks
python src/ragas_eval.py --limit 5  # quick trial
python src/ragas_eval.py            # all 30 questions (slow on a laptop)
```

Results: `data/evaluation/ragas_report.txt` and `ragas_results.csv`.
