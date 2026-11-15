"""
Query expansion (HyDE-style), applied before retrieval.

The user's question is phrased in their own words, which may differ from
the wording in the document. To bridge that gap, we ask the LLM to write
a hypothetical answer, then embed both the question and the hypothetical
answer and retrieve with both. Results are merged by distance, so chunks
that match either phrasing are candidates.

If the LLM is unavailable, the expansion is skipped and retrieval falls
back to the question alone — expansion never blocks a query.
"""

from typing import List

from app.config import settings
from app.logging_config import get_logger
from app.services import llm_service

logger = get_logger(__name__)

_HYDE_PROMPT = (
    "Write a short, factual answer to the following question as if you were "
    "summarizing the source document. If you do not know the answer, write a "
    "plausible answer anyway. Do not explain yourself.\n\nQuestion: {question}\n\nAnswer:"
)


async def expand_query(question: str) -> List[str]:
    """
    Return the list of queries to embed: the original question plus, when
    enabled and available, a hypothetical answer from the LLM.
    """
    queries = [question]

    if not settings.query_expansion_enabled:
        return queries

    try:
        hypothetical = await llm_service.generate_answer(_HYDE_PROMPT.format(question=question))
        if hypothetical.strip():
            queries.append(hypothetical.strip())
            logger.info("query_expansion.expanded", extra={"queries": len(queries)})
    except Exception as exc:  # noqa: BLE001 — expansion is best-effort
        logger.warning("query_expansion.skipped", extra={"error": str(exc)})

    return queries