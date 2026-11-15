"""
Stage 6: Prompt construction.

Combines the retrieved chunks with the user's question into a single
prompt that instructs the LLM to answer only from the given context,
which is what keeps the system's answers grounded in the retrieved
evidence instead of the model's own memory.
"""

from typing import List, Dict, Any

from app.logging_config import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """Answer only from the provided context.

If the answer is not in the context, reply exactly:
"I could not find this information in the uploaded documents."

Do not guess.

When you use a piece of the context, cite it inline right after the claim
using the tag [Cite: <source> | Chunk: <index>] with the exact source file
and chunk index shown in the context. Cite every claim you make.

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_blocks = [
        f"[Source: {chunk['source']} | Chunk: {chunk['chunk_index']}]\n{chunk['content']}"
        for chunk in chunks
    ]
    context = "\n\n".join(context_blocks)
    logger.info(
        "prompt.build_prompt",
        extra={"question_length": len(question), "chunks_count": len(chunks)},
    )
    return _PROMPT_TEMPLATE.format(context=context, question=question)
