"""
ragas_eval.py - RAGAS evaluation of the Scholarship RAG (fully local, no API key).

RAGAS ("Retrieval-Augmented Generation Assessment") scores the QUALITY of the RAG
steps, which verdict accuracy / F1 cannot show:

  faithfulness                       Is every claim in the answer supported by the
                                     retrieved chunks? (catches hallucination)
  answer_relevancy                   Does the answer actually address the question?
  llm_context_precision_with_reference
                                     Are the useful chunks ranked at the top of the
                                     retrieved list? (retrieval ranking quality)
  context_recall                     Do the retrieved chunks contain the information
                                     needed for the correct answer? (retrieval coverage)

All scores are 0-1, higher is better. RAGAS uses a "judge" LLM to make these
judgements; here that is a local Ollama model.

Steps (from the project folder):
    python src/evaluation.py            # 1) answers + retrieved chunks -> evaluation_results.csv
    pip install -r requirements-ragas.txt
    python src/ragas_eval.py --limit 5  # 2) quick trial on 5 questions
    python src/ragas_eval.py            # 3) all 30 questions (slow on a laptop)

Outputs: data/evaluation/ragas_results.csv and data/evaluation/ragas_report.txt
"""

import argparse
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import pandas as pd
    from langchain_core.embeddings import Embeddings
    from langchain_ollama import ChatOllama
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from ragas.run_config import RunConfig
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing package ({e}).\nInstall the RAGAS extras first:\n"
             f"    pip install -r requirements-ragas.txt")

from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING = "BAAI/bge-small-en-v1.5"
METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
}


class LocalEmbeddings(Embeddings):
    """LangChain wrapper around a local sentence-transformers model (used by answer_relevancy)."""

    def __init__(self, model_name: str):
        self.model = model_name            # RAGAS reads `.model` as the model's NAME (a string)
        self._encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self._encoder.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self._encoder.encode([text], normalize_embeddings=True)[0].tolist()


def build_samples(rows):
    """Turn evaluation_results.csv rows into RAGAS samples."""
    samples = []
    for r in rows:
        profile = (r.get("student_profile") or "").strip()
        question = r["query"] if not profile else f"Student profile: {profile}\nQuestion: {r['query']}"
        samples.append({
            "user_input": question,
            "response": r["response"],
            "retrieved_contexts": json.loads(r["contexts"]),
            # Reference answer = the correct verdict + the clause that decides it
            "reference": f"The student is {r['ground_truth'].replace('_', ' ').lower()}. "
                         f"Deciding rule: {r.get('critical_clause', '')}",
        })
    return samples


def mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation with a local Ollama judge.")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--results", default=os.path.join(base, "data/evaluation/evaluation_results.csv"))
    parser.add_argument("--judge", default="qwen2.5:7b", help="Ollama model used as the RAGAS judge")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    parser.add_argument("--metrics", nargs="+", default=list(METRIC_NAMES), choices=list(METRIC_NAMES))
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    parser.add_argument("--workers", type=int, default=2, help="parallel judge calls")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per judge call")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if "contexts" not in df.columns:
        sys.exit("evaluation_results.csv has no 'contexts' column.\n"
                 "Re-run `python src/evaluation.py` with the updated code first.")
    df = df.fillna("")
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")

    judge = LangchainLLMWrapper(ChatOllama(model=args.judge, temperature=0, num_ctx=8192))
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings(args.embedding))
    available = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(strictness=1),  # 1 generated question: faster locally
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
    metrics = [available[m] for m in args.metrics]

    print(f"RAGAS: {len(rows)} questions x {len(metrics)} metrics, judge = {args.judge}")
    print("This makes many local LLM calls; expect several minutes per question on a laptop.")
    start = time.time()
    result = evaluate(
        dataset=EvaluationDataset.from_list(build_samples(rows)),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=args.timeout, max_retries=3, max_workers=args.workers),
        raise_exceptions=False,   # a failed judgement becomes NaN instead of stopping the run
        show_progress=True,
    )
    scores = result.to_pandas()
    minutes = (time.time() - start) / 60

    # Join scores back onto the question ids / difficulty / correctness
    keep = ["query_id", "difficulty", "demographic_tag", "ground_truth", "predicted", "correct"]
    out = pd.concat([df[keep].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    out_csv = os.path.join(base, "data/evaluation/ragas_results.csv")
    out.to_csv(out_csv, index=False)

    cols = [METRIC_NAMES[m] for m in args.metrics if METRIC_NAMES[m] in out.columns]
    correct_mask = out["correct"].astype(str).str.lower() == "true"
    lines = ["=" * 70, "  RAGAS EVALUATION REPORT", "=" * 70,
             f"  Judge LLM: {args.judge}   Embeddings: {args.embedding}",
             f"  Questions: {len(out)}   Run time: {minutes:.1f} min", "",
             f"  {'Metric':<40}{'Mean':>8}{'Scored':>9}"]
    for c in cols:
        m, n = mean(out[c].tolist())
        lines.append(f"  {c:<40}{m:>8.3f}{n:>6}/{len(out)}")
    lines += ["", "  BY DIFFICULTY"]
    for diff in sorted(out["difficulty"].unique()):
        sub = out[out["difficulty"] == diff]
        lines.append(f"  {diff:<16}" + "  ".join(f"{c.split('_')[0][:5]}={mean(sub[c].tolist())[0]:.2f}"
                                                  for c in cols))
    lines += ["", "  CORRECT vs WRONG VERDICTS (do wrong answers have weaker retrieval / grounding?)"]
    for label, sub in [("correct", out[correct_mask]), ("wrong", out[~correct_mask])]:
        lines.append(f"  {label:<16}" + "  ".join(f"{c.split('_')[0][:5]}={mean(sub[c].tolist())[0]:.2f}"
                                                  for c in cols))
    lines += ["", "  Unscored cells (NaN) = the judge's reply could not be parsed; a larger",
              "  judge model (e.g. qwen2.5:14b) gives fewer of these.", "=" * 70]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(base, "data/evaluation/ragas_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved: {out_csv}\n       data/evaluation/ragas_report.txt")


if __name__ == "__main__":
    main()
