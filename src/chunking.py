"""
chunking.py - Semantic chunking strategy for scholarship documents.
Preserves section headers and keeps exception clauses with parent context.
"""

import re


def semantic_chunk(text, source_name, min_size=200, max_size=1000):
    """
    Chunk documents by semantic sections while preserving exception clauses.

    Strategy:
    - Split by numbered section headers (e.g., "3.3.1", "C3.4", "5.4")
    - Keep each section as its own chunk where possible
    - Merge small adjacent sections to stay above min_size
    - Split oversized sections at sentence boundaries
    """
    # Pattern to match section headers across all 3 documents
    header_pattern = r'(?:^|\n)\s*(?:Section\s+\d+|\d+(?:\.\d+)+|[A-C]\d+(?:\.\d+)?|Part\s+[A-C]|\d+\.\s+\w+)\s*[\:\.\-]?\s*'

    # Split text by headers
    parts = re.split(f'({header_pattern})', text)

    chunks = []
    current_chunk = ""
    current_meta = source_name

    for part in parts:
        if not part or not part.strip():
            continue

        # Check if this part is a header
        if re.match(header_pattern.strip(), part.strip(), re.IGNORECASE):
            # Save previous chunk if it exists and is large enough
            if current_chunk and len(current_chunk.strip()) > min_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source_name,
                    "metadata": {"section": current_meta}
                })
                current_chunk = ""
            current_meta = part.strip()
            current_chunk += part + "\n"
        else:
            current_chunk += part + "\n"

        # If chunk exceeds max_size, keep splitting at sentence boundaries.
        # (Was `if`, which split only once and left chunks of 5,000+ characters -
        #  far beyond what the embedding model reads.)
        while len(current_chunk) > max_size:
            # Find last sentence boundary before max_size
            split_point = current_chunk.rfind('. ', min_size, max_size)
            if split_point == -1:
                split_point = current_chunk.rfind('\n', min_size, max_size)
            if split_point == -1:
                split_point = max_size

            head = current_chunk[:split_point + 1]
            chunks.append({
                "text": head.strip(),
                "source": source_name,
                "metadata": {"section": current_meta}
            })
            # Overlap: repeat the last sentence so a rule and its exception
            # ("... unless / except ...") are never separated.
            last_sentence = re.split(r'(?<=[.!?])\s+|\n+', head.strip())[-1]
            if len(last_sentence) > 300:
                last_sentence = ""
            rest = current_chunk[split_point + 1:].lstrip()
            candidate = current_meta + "\n" + last_sentence + " " + rest
            # Guarantee progress: drop the overlap if it would stop the chunk shrinking
            if len(candidate) >= len(current_chunk) or len(candidate) - len(rest) >= max_size // 2:
                candidate = current_meta + "\n" + rest
            if len(candidate) >= len(current_chunk):  # cannot shrink further; stop splitting
                break
            current_chunk = candidate

    # Add remaining chunk
    if current_chunk and len(current_chunk.strip()) > 50:
        chunks.append({
            "text": current_chunk.strip(),
            "source": source_name,
            "metadata": {"section": current_meta}
        })

    return chunks


def load_and_chunk_all(kb1_path, kb2_path, kb3_path):
    """Load all three knowledge base documents and chunk them."""
    all_chunks = []

    with open(kb1_path, 'r', encoding='utf-8') as f:
        kb1_text = f.read()
    all_chunks.extend(semantic_chunk(kb1_text, "KB-1_RMIT_Terms"))

    with open(kb2_path, 'r', encoding='utf-8') as f:
        kb2_text = f.read()
    all_chunks.extend(semantic_chunk(kb2_text, "KB-2_RMIT_RTP_Policy"))

    with open(kb3_path, 'r', encoding='utf-8') as f:
        kb3_text = f.read()
    all_chunks.extend(semantic_chunk(kb3_text, "KB-3_ARC_Guidelines"))

    return all_chunks


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks = load_and_chunk_all(
        os.path.join(base, "data/raw/kb1_rmit_scholarship_terms.txt"),
        os.path.join(base, "data/raw/kb2_rmit_rtp_policy.txt"),
        os.path.join(base, "data/raw/kb3_arc_discovery_guidelines.txt")
    )
    print(f"Total chunks created: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} [{c['source']}] ---")
        print(c['text'][:300] + "...")
