"""
retrieval_eval.py - Compare retrieval setups WITHOUT calling the language model.

For each gold-standard question it checks whether the chunk containing the
question's critical clause was retrieved, and at what rank.

Metrics
  Hit@k : share of questions whose critical clause is in the top k chunks
  MRR   : Mean Reciprocal Rank = average of 1 / (rank of the first correct chunk);
          1.0 = always ranked first, 0 = never retrieved

Usage (from the project folder):
    python src/retrieval_eval.py                         # dense vs hybrid, bge-small
    python src/retrieval_eval.py --embedding all-MiniLM-L6-v2
    python src/retrieval_eval.py --modes dense --k 6
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunking import load_and_chunk_all
from evaluation import load_test_cases
from rag_pipeline import ScholarshipRAG, DEFAULT_EMBEDDING_MODEL


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def clause_key(critical_clause: str) -> str:
    """First part of the critical clause, used to recognise the right chunk."""
    return _norm(critical_clause.split("...")[0])[:60]


def first_hit_rank(docs, key):
    for rank, doc in enumerate(docs, 1):
        if key and key in _norm(doc):
            return rank
    return None


def evaluate_mode(chunks, cases, mode, embedding, k, use_profile):
    rag = ScholarshipRAG(chunks, embedding_model=embedding, retrieval_mode=mode,
                         collection_name=f"retrieval_eval_{mode}")
    all_text = _norm(" ".join(rag.texts))
    ranks, rows = [], []
    for case in cases:
        key = clause_key(case.get("critical_clause", ""))
        if not key or key not in all_text:
            continue  # clause text not found verbatim in the KB -> can't score retrieval
        profile = case.get("student_profile", "") if use_profile else ""
        docs, _ = rag.retrieve(case["natural_language_query"], n_results=k, profile=profile)
        rank = first_hit_rank(docs, key)
        ranks.append(rank)
        rows.append((case["query_id"], rank))
    n = len(ranks)
    hit = lambda cut: sum(1 for r in ranks if r and r <= cut) / n if n else 0.0
    mrr = sum(1.0 / r for r in ranks if r) / n if n else 0.0
    return {"mode": mode, "n": n, "hit1": hit(1), "hit3": hit(3), "hitk": hit(k), "mrr": mrr,
            "misses": [q for q, r in rows if not r]}


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation (no LLM).")
    parser.add_argument("--modes", nargs="+", default=["dense", "hybrid"], choices=["dense", "hybrid"])
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks = load_and_chunk_all(
        os.path.join(base, "data/raw/kb1_rmit_scholarship_terms.txt"),
        os.path.join(base, "data/raw/kb2_rmit_rtp_policy.txt"),
        os.path.join(base, "data/raw/kb3_arc_discovery_guidelines.txt"),
    )
    cases = load_test_cases(os.path.join(base, "data/evaluation/gold_standard.csv"))

    results = [evaluate_mode(chunks, cases, m, args.embedding, args.k, not args.no_profile)
               for m in args.modes]

    lines = ["=" * 70, "  RETRIEVAL EVALUATION (no LLM)", "=" * 70,
             f"  Embedding model: {args.embedding}   top_k={args.k}   "
             f"profile={'off' if args.no_profile else 'on'}", "",
             f"  {'Mode':<10}{'Scored':>8}{'Hit@1':>9}{'Hit@3':>9}{f'Hit@{args.k}':>9}{'MRR':>9}"]
    for r in results:
        lines.append(f"  {r['mode']:<10}{r['n']:>8}{r['hit1']:>9.1%}{r['hit3']:>9.1%}"
                     f"{r['hitk']:>9.1%}{r['mrr']:>9.3f}")
    lines.append("")
    for r in results:
        lines.append(f"  {r['mode']} missed: {', '.join(r['misses']) or 'none'}")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print("\n" + report)

    out = os.path.join(base, "data/evaluation/retrieval_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
