"""Prompts for question generation and query transformation."""

QUESTION_PROMPT = """\
You write evaluation questions for a document search system. Given one passage, write up to \
{n} questions that a real user might ask and that this passage answers.

Rules:
- Each question must be answerable from the passage alone, and specific enough that the \
passage is clearly the right place to look (no "What does the document say?").
- Phrase questions the way a user would, without copying long phrases from the passage, so \
the search has to work.
- For each question, give `evidence`: a short exact quote (one sentence or less, copied \
character for character) from the passage that answers it.
- If the passage has no useful facts (a table of contents, boilerplate), return no questions.
"""

REWRITE_PROMPT = """\
Rewrite the user's question into a search query for a document index that uses both keyword \
and semantic search: keep the key entities, add likely synonyms and the specific terms a \
document answering it would use. Return only the query.
"""

HYDE_PROMPT = """\
Write a short passage (2-4 sentences) that would plausibly appear in a document answering the \
question, in the style of documentation. Invent plausible specifics if needed; it is only used \
to find similar real passages. Return only the passage.
"""
