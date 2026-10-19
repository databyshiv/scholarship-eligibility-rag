"""
rag_pipeline.py - Core RAG pipeline for Scholarship Eligibility Checker.
Uses ChromaDB for vector storage and Ollama for local LLM inference.

Retrieval: dense (embedding) search by default; BM25 hybrid is optional.

v2 improvements (for conditional / exception questions):
  1. Student profile is passed to retrieval AND to the LLM (it was ignored before).
  2. Hybrid retrieval: dense embeddings + BM25 keyword search, merged with
     Reciprocal Rank Fusion. Catches exact terms like "DECRA", "EFTSL", "75%".
  3. Optional filter to one knowledge base (e.g. only KB-3 for DECRA questions).
  4. Step-by-step prompt: facts -> rules -> check each condition/exception
     -> verdict LAST (so the model reasons before committing).
  5. Robust verdict parsing shared by evaluation.py and the Streamlit app.
"""

import math
import re
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

try:
    import ollama
except ImportError:
    ollama = None
    print("WARNING: ollama not installed. Install with: pip install ollama")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("WARNING: sentence-transformers not installed.")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    print("WARNING: chromadb not installed.")


# Map the short KB ids used in gold_standard.csv to chunk source names
KB_SOURCES = {
    "KB-1": "KB-1_RMIT_Terms",
    "KB-2": "KB-2_RMIT_RTP_Policy",
    "KB-3": "KB-3_ARC_Guidelines",
}

VERDICTS = ["ELIGIBLE", "INELIGIBLE", "CANNOT_DETERMINE"]


# ----------------------------------------------------------------------
# Verdict parsing (shared)
# ----------------------------------------------------------------------
def extract_verdict(response: str) -> str:
    """
    Extract ELIGIBLE / INELIGIBLE / CANNOT_DETERMINE from an LLM response.

    Uses the LAST 'VERDICT:' line (the prompt asks for the verdict at the end),
    tolerates markdown (**VERDICT:** ...), brackets and 'CANNOT DETERMINE'.
    """
    if not response:
        return "UNPARSEABLE"
    text = response.upper().replace("*", "").replace("#", "")
    matches = re.findall(r"VERDICT\s*[:\-]?\s*\[?\s*(CANNOT[\s_]+DETERMINE|INELIGIBLE|ELIGIBLE)", text)
    if matches:
        v = matches[-1]
        return "CANNOT_DETERMINE" if v.startswith("CANNOT") else v

    # Fallback: first verdict word that appears as a whole word
    m = re.search(r"\b(CANNOT[\s_]+DETERMINE|INELIGIBLE|ELIGIBLE)\b", text)
    if m:
        v = m.group(1)
        return "CANNOT_DETERMINE" if v.startswith("CANNOT") else v
    return "UNPARSEABLE"


# ----------------------------------------------------------------------
# Tiny BM25 (no extra dependency)
# ----------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9$%.]+")
_STOP = set("a an the of to and or in on for is are be i my me can will do does it this that with at by as "
            "if from have has not no your you am was were get".split())


def _tokenize(text: str) -> List[str]:
    return [t.strip(".") for t in _TOKEN.findall(text.lower()) if t.strip(".") and t not in _STOP]


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(d) for d in docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> List[float]:
        q = _tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


# ----------------------------------------------------------------------
# Query expansion: map everyday wording to the vocabulary used in the policies
# ----------------------------------------------------------------------
_EXPANSIONS = [
    (r"\b(earn|earns|earning|earnings|salary|wage|wages|job|jobs|work|working|paid|pay|employment|employed)\b",
     "income from another source living costs employment"),
    (r"\b(kid|kids|child|children|baby|parent|parental|maternity|carer|caring)\b",
     "parental leave carer career interruption"),
    (r"\b(break|pause|time off|leave|sick|illness|medical)\b", "leave suspension"),
    (r"\b(lose|losing|lost|cancel|cancelled|stop|stopped|cut|revoke|terminated?)\b",
     "terminate termination cease"),
    (r"\b(fail|failed|failing|unsatisfactory|poor progress|review)\b", "unsatisfactory academic progress"),
    (r"\b(move|moving|transfer|transferring|switch|another uni|other uni)\b", "transfer"),
    (r"\b(applied|apply|nominated|nomination|rejected|withdrew|withdrawn)\b", "nominated previous occasions"),
    (r"\b(scholarship|stipend|rtp)\b", "RTP stipend scholarship"),
]


def expand_query(query: str) -> str:
    """Append policy terms for everyday words found in the question (used for search only)."""
    extra = [terms for pattern, terms in _EXPANSIONS if re.search(pattern, query, re.I)]
    return f"{query} {' '.join(extra)}".strip()


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise eligibility checker for scholarships and research grants. "
    "You only use the provided documents, you never invent rules, and you check every "
    "condition and exception against the student's facts before giving a verdict."
)

PROMPT_TEMPLATE = """Decide the student's eligibility using ONLY the context documents below.

VERDICT MEANINGS
- ELIGIBLE: the documents allow what the student is asking for. This includes answers like
  "yes, you can keep it", "this does not affect you", or "you receive it for the remaining period".
- INELIGIBLE: a rule in the documents excludes the student, or ends/terminates what they are asking about.
- CANNOT_DETERMINE: the answer depends on a specific fact that is NOT in the student's profile or
  question and NOT in the documents (for example a date only published on an external website, or a
  decision left to someone's discretion). Name the missing fact. Do not use CANNOT_DETERMINE just
  because the question is short - if a rule clearly applies, decide.

HOW TO REASON
1. FACTS: list every fact about the student from the profile and question (citizenship, degree,
   dates, durations, amounts, number of previous applications, etc.).
2. RULES: find the rules that apply to the scholarship/grant being asked about. Ignore rules about
   a different scholarship or grant scheme. Include every exception ("unless", "except",
   "provided that", "only if", "does not apply", "is not to be taken into account").
3. CHECK: test each rule against the facts one at a time. Do any arithmetic explicitly
   (compare numbers, dates and counts). For a limit such as "on two previous occasions", the student
   is only excluded if they have REACHED the limit. Then check whether an exception overrides the rule.
4. Decide, and quote the single clause that decides the answer.

CONTEXT DOCUMENTS:
{context}

STUDENT PROFILE: {profile}

STUDENT QUESTION: {query}

Respond in exactly this format:
FACTS: <bullet list>
ANALYSIS: <rule-by-rule check, including exceptions and arithmetic>
CITATION: "<exact quote from the documents>" (Source, Section)
VERDICT: <ELIGIBLE or INELIGIBLE or CANNOT_DETERMINE>
"""


# Some embedding models are trained with an instruction in front of search queries.
# (Documents are embedded without it.)
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RETRIEVAL_MODES = ("dense", "hybrid")


class ScholarshipRAG:
    """
    Retrieval-Augmented Generation pipeline for checking scholarship eligibility.

    retrieval_mode:
      "dense"  (default) - embedding (vector) search only, via ChromaDB.
      "hybrid"           - dense search + BM25 keyword search, merged with
                           Reciprocal Rank Fusion. Kept for comparison.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = 'qwen2.5:7b',
        collection_name: str = "scholarship_kb",
        retrieval_mode: str = "dense",
        hybrid: Optional[bool] = None,   # old flag, kept so older scripts still work
    ):
        if SentenceTransformer is None or chromadb is None or ollama is None:
            raise RuntimeError(
                "Missing dependencies. Install: pip install sentence-transformers chromadb ollama"
            )
        if hybrid is not None:
            retrieval_mode = "hybrid" if hybrid else "dense"
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}")

        self.chunks = chunks
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.retrieval_mode = retrieval_mode
        self.query_instruction = QUERY_INSTRUCTIONS.get(embedding_model, "")

        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)

        print(f"Initializing ChromaDB collection: {collection_name}...")
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        # cosine distance suits (normalised) sentence-transformer embeddings
        self.collection = self.client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

        print(f"Embedding {len(chunks)} chunks...")
        self.texts = [c["text"] for c in chunks]
        self.metas = [{"source": c["source"], **c.get("metadata", {})} for c in chunks]
        embeddings = self.embed(self.texts, is_query=False)
        self.ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(embeddings=embeddings, documents=self.texts,
                            ids=self.ids, metadatas=self.metas)

        self.bm25 = BM25(self.texts) if retrieval_mode == "hybrid" else None
        print(f"RAG pipeline ready! (retrieval: {retrieval_mode})")

    @property
    def hybrid(self) -> bool:
        return self.retrieval_mode == "hybrid"

    def embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        """Embed texts; search queries get the model's query instruction (if it has one)."""
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        vectors = self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=not is_query)
        return vectors.tolist()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, n_results: int = 6, profile: str = "",
                 source_filter: Optional[str] = None,
                 with_scores: bool = False):
        """
        Retrieve the top-k chunks.

        Dense retrieval: the question and (question + profile) are each embedded and
        searched in ChromaDB. The question counts fully, question + profile at half
        weight, because generic profile words ("Domestic", "PhD") otherwise pull in
        irrelevant sections. The two rankings are merged with weighted Reciprocal
        Rank Fusion.

        Hybrid mode additionally merges BM25 keyword rankings (with query expansion).

        source_filter: 'KB-1' / 'KB-2' / 'KB-3' (or a full source name) to restrict the search.
        with_scores: also return each chunk's cosine similarity to the question (0-1).
        """
        source = KB_SOURCES.get(source_filter, source_filter) if source_filter else None
        allowed = [i for i, m in enumerate(self.metas) if source is None or m["source"] == source]
        if not allowed:
            allowed = list(range(len(self.metas)))
        allowed_set = set(allowed)
        pool = min(len(allowed), max(n_results * 3, 15))

        p_text = f"{query} {profile}".strip() if profile.strip() else ""
        similarity: Dict[int, float] = {}

        def dense_rank(text: str, record: bool = False) -> List[int]:
            kwargs = {"query_embeddings": self.embed([text], is_query=True), "n_results": pool}
            if source and len(allowed) < len(self.metas):
                kwargs["where"] = {"source": source}
            res = self.collection.query(**kwargs)
            ranked = [int(i.split("_")[1]) for i in res["ids"][0]]
            if record:  # cosine similarity = 1 - cosine distance
                for idx, dist in zip(ranked, res["distances"][0]):
                    similarity[idx] = round(1.0 - float(dist), 4)
            return ranked

        # (ranking, weight) pairs
        rankings = [(dense_rank(query, record=True), 1.0)]
        if p_text:
            rankings.append((dense_rank(p_text), 0.5))
        if self.hybrid:
            q_expanded = expand_query(query)
            scores = self.bm25.scores(q_expanded)
            rankings.append((sorted(allowed, key=lambda i: scores[i], reverse=True)[:pool], 1.0))
            if p_text:
                scores = self.bm25.scores(f"{q_expanded} {profile}")
                rankings.append((sorted(allowed, key=lambda i: scores[i], reverse=True)[:pool], 0.5))

        # Weighted Reciprocal Rank Fusion
        fused: Dict[int, float] = {}
        for rank_list, weight in rankings:
            for r, idx in enumerate(rank_list):
                if idx in allowed_set:
                    fused[idx] = fused.get(idx, 0.0) + weight / (60 + r)
        top = sorted(fused, key=fused.get, reverse=True)[:n_results]

        docs = [self.texts[i] for i in top]
        metas = [self.metas[i] for i in top]
        if with_scores:
            return docs, metas, [similarity.get(i) for i in top]
        return docs, metas

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, query: str, retrieved_docs: List[str], retrieved_metas: List[Dict],
                 profile: str = "") -> str:
        context_parts = []
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), 1):
            section = meta.get('section', 'Unknown Section')
            source = meta.get('source', 'Unknown Source')
            context_parts.append(f"[Document {i} | Source: {source} | Section: {section}]\n{doc}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            profile=profile.strip() or "Not provided (use only the facts in the question).",
            query=query,
        )
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                options={
                    'temperature': 0.0,   # deterministic -> repeatable evaluation
                    'num_predict': 1000,  # room for step-by-step reasoning
                    'num_ctx': 8192,      # make sure all retrieved chunks fit
                    'seed': 42,
                },
            )
            return response['message']['content']
        except Exception as e:
            return f"ERROR: Could not generate response. {str(e)}"

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def answer(self, query: str, n_results: int = 6, profile: str = "",
               source_filter: Optional[str] = None) -> str:
        docs, metas = self.retrieve(query, n_results, profile, source_filter)
        return self.generate(query, docs, metas, profile)

    def answer_with_sources(self, query: str, n_results: int = 6, profile: str = "",
                            source_filter: Optional[str] = None) -> Dict[str, Any]:
        docs, metas, scores = self.retrieve(query, n_results, profile, source_filter, with_scores=True)
        response = self.generate(query, docs, metas, profile)
        return {
            "query": query,
            "profile": profile,
            "response": response,
            "verdict": extract_verdict(response),
            "sources": [{"text": d, "metadata": m, "similarity": sc}
                        for d, m, sc in zip(docs, metas, scores)],
        }


if __name__ == "__main__":
    from chunking import load_and_chunk_all
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks = load_and_chunk_all(
        os.path.join(base, "data/raw/kb1_rmit_scholarship_terms.txt"),
        os.path.join(base, "data/raw/kb2_rmit_rtp_policy.txt"),
        os.path.join(base, "data/raw/kb3_arc_discovery_guidelines.txt")
    )
    rag = ScholarshipRAG(chunks)

    test_query = "I am a single parent with a preschool child. Can I get a part-time RTP stipend?"
    print(f"\nQuery: {test_query}")
    print("=" * 60)
    print(rag.answer(test_query, profile="Domestic PhD student"))
