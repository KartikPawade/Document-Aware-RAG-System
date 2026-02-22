"""
ingestion/chunkers/utils.py
────────────────────────────
Shared chunking utilities used by all format plugins.

Functions:
- estimate_tokens          : fast token count approximation
- semantic_chunk           : sentence-aware fixed-size chunking
- heading_based_chunk      : group text blocks by heading into sections
- create_parent_child_chunks : create parent + child Chunk objects with parent_id linking
"""
from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from src.core.models import Chunk, ContentType


# ─────────────────────────────────────────────────────────────────────────────
# Token Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Fast approximation: 1 token ≈ 4 characters (matches GPT tokenization closely).
    For precise counts, use tiktoken — but this is sufficient for chunking decisions.
    """
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-Aware Chunking
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a regex that handles common abbreviations.
    Falls back to splitting on double-newlines if no sentence boundaries found.
    """
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    # Further split on paragraph breaks
    result = []
    for s in sentences:
        parts = s.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result if result else [text]


def semantic_chunk(
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Sentence-aware chunking: never cuts a sentence mid-way.
    Produces chunks of approximately target_tokens with overlap_tokens of overlap.

    Algorithm:
    1. Split text into sentences
    2. Greedily accumulate sentences until target_tokens is reached
    3. Slide window back by overlap_tokens for the next chunk
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = estimate_tokens(sentence)

        if current_tokens + s_tokens > target_tokens and current:
            # Flush current chunk
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Slide back: keep last `overlap_tokens` worth of sentences
            overlap_sentences: List[str] = []
            overlap_total = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if overlap_total + prev_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, prev)
                    overlap_total += prev_tokens
                else:
                    break
            current = overlap_sentences
            current_tokens = overlap_total

        current.append(sentence)
        current_tokens += s_tokens

    # Flush remainder
    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Heading-Based Sectioning
# ─────────────────────────────────────────────────────────────────────────────

def heading_based_chunk(
    text_blocks,           # List[TextBlock]
    target_tokens: int = 1500,
) -> List[Tuple[Optional[str], str]]:
    """
    Group TextBlock objects into sections by heading.
    Returns a list of (heading, section_text) tuples.

    Each section accumulates blocks until target_tokens is reached,
    then starts a new section under the same heading (overflow split).
    """
    sections: List[Tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    current_parts: List[str] = []
    current_tokens: int = 0

    def flush():
        nonlocal current_parts, current_tokens
        if current_parts:
            text = "\n\n".join(current_parts).strip()
            if text:
                sections.append((current_heading, text))
            current_parts = []
            current_tokens = 0

    for block in text_blocks:
        block_tokens = estimate_tokens(block.text)

        # New heading → flush and start fresh section
        if block.heading and block.heading != current_heading:
            flush()
            current_heading = block.heading

        # Overflow: current section is full → flush and continue under same heading
        if current_tokens + block_tokens > target_tokens and current_parts:
            flush()

        current_parts.append(block.text)
        current_tokens += block_tokens

    flush()
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Parent-Child Chunk Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_parent_child_chunks(
    parent_text: str,
    heading: Optional[str],
    source_metadata: dict,
    child_target_tokens: int = 400,
    child_overlap_tokens: int = 50,
    chunk_index_offset: int = 0,
) -> Tuple[Chunk, List[Chunk]]:
    """
    Create a parent Chunk and its list of child Chunks.

    Parent:
    - Contains the full section text (up to ~1500 tokens)
    - parent_id = None (it IS the parent)

    Children:
    - Semantically chunked sub-sections of the parent (~400 tokens each)
    - parent_id = parent.chunk_id
    - Used for ANN retrieval (small chunks score better)
    - Context expansion at query time fetches the parent for the LLM

    Returns:
        (parent_chunk, [child_chunk_1, child_chunk_2, ...])
    """
    parent_id = str(uuid.uuid4())

    parent = Chunk(
        text=parent_text,
        chunk_id=parent_id,
        parent_id=None,                                     # Parent has no parent
        source_id=source_metadata.get("source_id", ""),
        source_url=source_metadata.get("source_url", ""),
        section_heading=heading,
        content_type=ContentType.PROSE,
        token_count=estimate_tokens(parent_text),
    )

    # Generate children by semantic chunking the parent text
    child_texts = semantic_chunk(
        parent_text,
        target_tokens=child_target_tokens,
        overlap_tokens=child_overlap_tokens,
    )

    children: List[Chunk] = []
    for i, child_text in enumerate(child_texts):
        child = Chunk(
            text=child_text,
            parent_id=parent_id,                           # Link to parent
            source_id=source_metadata.get("source_id", ""),
            source_url=source_metadata.get("source_url", ""),
            section_heading=heading,
            content_type=ContentType.PROSE,
            chunk_index=chunk_index_offset + i,
            total_chunks=len(child_texts),
            token_count=estimate_tokens(child_text),
        )
        children.append(child)

    # If no children were produced (very short text), use parent itself as the only chunk
    if not children:
        parent.chunk_index = chunk_index_offset
        children = []

    return parent, children