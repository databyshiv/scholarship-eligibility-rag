"""
evaluation.py - Test-driven evaluation framework for the Scholarship RAG.
Implements quantitative metrics: accuracy, % unanswered, fairness, and per-difficulty breakdown.
"""

import os
import sys
import re
import csv
import json
from typing import Dict, List, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunking import load_and_chunk_all
from rag_pipeline import ScholarshipRAG, extract_verdict, DEFAULT_EMBEDDING_MODEL
from metrics import per_class, LABELS


def load_test_cases(csv_path: str) -> List[Dict[str, str]]:
    """Load gold-standard test cases from CSV."""
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


def evaluate_pipeline(rag: ScholarshipRAG, test_cases: List[Dict[str, str]], output_dir: str = None,
                      use_profile: bool = True, filter_kb: bool = False, n_results: int = 6) -> Dict[str, Any]:
    """
    Run the full evaluation suite.

    Returns:
        Dictionary with metrics, results DataFrame-like list, and report text.
    """
    results = []

    print(f"\nRunning evaluation on {len(test_cases)} test cases...")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        query_id = case['query_id']
        query = case['natural_language_query']
        ground_truth = case['ground_truth'].strip().upper()
        difficulty = case['difficulty']
        demographic = case['demographic_tag']

        print(f"[{i}/{len(test_cases)}] {query_id} ({difficulty}) | {query[:50]}...")

        # Run RAG
        profile = case.get('student_profile', '') if use_profile else ''
        source_filter = case.get('target_scholarship') if filter_kb else None
        out = rag.answer_with_sources(query, n_results=n_results, profile=profile,
                                      source_filter=source_filter)
        response = out['response']
        contexts = [src['text'] for src in out['sources']]
        predicted = extract_verdict(response)

        correct = (predicted == ground_truth)

        results.append({
            'query_id': query_id,
            'query': query,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': correct,
            'difficulty': difficulty,
            'demographic_tag': demographic,
            'response': response,
            'critical_clause': case.get('critical_clause', ''),
            'student_profile': profile,
            # retrieved chunks, saved for RAGAS (ragas_eval.py) and retrieval checks
            'contexts': json.dumps(contexts, ensure_ascii=False),
        })

    # Calculate metrics
    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    overall_accuracy = correct_count / total if total > 0 else 0
    unanswered_count = sum(1 for r in results if r['predicted'] == 'CANNOT_DETERMINE')
    unanswered_pct = unanswered_count / total if total > 0 else 0
    unparseable_count = sum(1 for r in results if r['predicted'] == 'UNPARSEABLE')

    # By difficulty
    difficulty_metrics = {}
    for diff in ['SIMPLE', 'CONDITIONAL', 'EXCEPTION_TRAP']:
        subset = [r for r in results if r['difficulty'] == diff]
        if subset:
            acc = sum(1 for r in subset if r['correct']) / len(subset)
            difficulty_metrics[diff] = {
                'count': len(subset),
                'accuracy': acc,
                'correct': sum(1 for r in subset if r['correct'])
            }

    # By demographic (fairness)
    demographic_metrics = {}
    for demo in set(r['demographic_tag'] for r in results):
        subset = [r for r in results if r['demographic_tag'] == demo]
        if subset:
            acc = sum(1 for r in subset if r['correct']) / len(subset)
            demographic_metrics[demo] = {
                'count': len(subset),
                'accuracy': acc,
                'correct': sum(1 for r in subset if r['correct'])
            }

    # By knowledge base
    kb_metrics = {}
    for kb in ['KB-1', 'KB-2', 'KB-3']:
        subset = [r for r in results if r['query_id'].startswith(kb.replace('KB-', 'Q0')) or 
                  r['query_id'].startswith(kb.replace('KB-', 'Q'))]
        # Better: map query IDs to KB
        kb_map = {'Q001-Q010': 'KB-1', 'Q011-Q020': 'KB-2', 'Q021-Q030': 'KB-3'}

    # F1 scores (per verdict class)
    gts = [r['ground_truth'] for r in results]
    preds = [r['predicted'] for r in results]
    class_stats = {l: per_class(gts, preds, l) for l in LABELS if l in gts or l in preds}
    support = sum(v[3] for v in class_stats.values()) or 1
    macro_f1 = sum(v[2] for v in class_stats.values()) / max(len(class_stats), 1)
    weighted_f1 = sum(v[2] * v[3] for v in class_stats.values()) / support

    # Build report
    report_lines = [
        "=" * 70,
        "  SCHOLARSHIP RAG - TEST-DRIVEN EVALUATION REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"LLM Model: {rag.llm_model}",
        f"Embedding Model: {rag.embedding_model_name}",
        f"Settings: profile={'on' if use_profile else 'off'}, kb_filter={'on' if filter_kb else 'off'}, "
        f"retrieval={getattr(rag, 'retrieval_mode', 'dense')}, top_k={n_results}",
        "",
        "-" * 70,
        "  OVERALL METRICS",
        "-" * 70,
        f"  Total Test Cases:        {total}",
        f"  Overall Accuracy:        {overall_accuracy:.1%}",
        f"  Correct Predictions:     {correct_count}/{total}",
        f"  % Unanswered:            {unanswered_pct:.1%} ({unanswered_count}/{total})",
        f"  Unparseable Responses:   {unparseable_count}/{total}",
        f"  Macro F1:                {macro_f1:.3f}",
        f"  Weighted F1:             {weighted_f1:.3f}",
        "",
        "-" * 70,
        "  BY DIFFICULTY (Conditional Logic Precision)",
        "-" * 70,
    ]

    for diff in ['SIMPLE', 'CONDITIONAL', 'EXCEPTION_TRAP']:
        if diff in difficulty_metrics:
            m = difficulty_metrics[diff]
            report_lines.append(
                f"  {diff:20s}  {m['accuracy']:.1%}  ({m['correct']}/{m['count']})"
            )

    report_lines.extend([
        "",
        "-" * 70,
        "  BY DEMOGRAPHIC (Fairness Analysis)",
        "-" * 70,
    ])

    for demo in sorted(demographic_metrics.keys()):
        m = demographic_metrics[demo]
        report_lines.append(
            f"  {demo:20s}  {m['accuracy']:.1%}  ({m['correct']}/{m['count']})"
        )

    report_lines.extend([
        "",
        "-" * 70,
        "  FAILED CASES (Ground Truth != Predicted)",
        "-" * 70,
    ])

    failed = [r for r in results if not r['correct']]
    if failed:
        for r in failed:
            report_lines.append(
                f"  {r['query_id']} | GT={r['ground_truth']:15s} | Pred={r['predicted']:15s} | {r['query'][:55]}..."
            )
    else:
        report_lines.append("  NONE — Perfect score!")

    report_lines.extend([
        "",
        "-" * 70,
        "  EXCEPTION TRAP ANALYSIS",
        "-" * 70,
    ])

    exception_cases = [r for r in results if r['difficulty'] == 'EXCEPTION_TRAP']
    exception_correct = sum(1 for r in exception_cases if r['correct'])
    report_lines.append(
        f"  Exception Trap Accuracy: {exception_correct}/{len(exception_cases)} "
        f"({exception_correct/len(exception_cases):.1%})" if exception_cases else "  N/A"
    )

    for r in exception_cases:
        status = "PASS" if r['correct'] else "FAIL"
        report_lines.append(f"  [{status}] {r['query_id']}: {r['query'][:60]}...")

    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    # Save results to CSV if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "evaluation_results.csv")
        with open(results_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query_id', 'query', 'ground_truth', 
                                                     'predicted', 'correct', 'difficulty',
                                                     'demographic_tag', 'response', 'critical_clause',
                                                     'student_profile', 'contexts'])
            writer.writeheader()
            writer.writerows(results)

        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nResults saved to: {results_path}")
        print(f"Report saved to: {report_path}")

    return {
        'metrics': {
            'overall_accuracy': overall_accuracy,
            'unanswered_pct': unanswered_pct,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'total_cases': total,
            'correct_count': correct_count,
            'difficulty': difficulty_metrics,
            'demographic': demographic_metrics
        },
        'results': results,
        'report': report
    }


def main():
    """Main entry point for running evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the gold-standard evaluation.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name (e.g. llama3.1, qwen2.5:7b)")
    parser.add_argument("--k", type=int, default=6, help="Number of chunks to retrieve")
    parser.add_argument("--no-profile", action="store_true", help="Do not send student_profile (old behaviour)")
    parser.add_argument("--filter-kb", action="store_true", help="Only search the target knowledge base")
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="dense",
                        help="dense = embedding search only (default); hybrid = dense + BM25")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING_MODEL,
                        help="sentence-transformers model, e.g. BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    chunks = load_and_chunk_all(
        os.path.join(base_dir, "data/raw/kb1_rmit_scholarship_terms.txt"),
        os.path.join(base_dir, "data/raw/kb2_rmit_rtp_policy.txt"),
        os.path.join(base_dir, "data/raw/kb3_arc_discovery_guidelines.txt")
    )
    print(f"Loaded {len(chunks)} chunks from 3 documents")

    rag = ScholarshipRAG(chunks, llm_model=args.model, embedding_model=args.embedding,
                         retrieval_mode=args.retrieval)

    test_cases = load_test_cases(os.path.join(base_dir, "data/evaluation/gold_standard.csv"))
    print(f"Loaded {len(test_cases)} test cases")

    output_dir = os.path.join(base_dir, "data/evaluation")
    eval_result = evaluate_pipeline(
        rag, test_cases, output_dir=output_dir,
        use_profile=not args.no_profile, filter_kb=args.filter_kb, n_results=args.k,
    )
    print("\n" + eval_result['report'])
    return eval_result


if __name__ == "__main__":
    main()
