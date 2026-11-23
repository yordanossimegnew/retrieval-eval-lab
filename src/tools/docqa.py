"""Import the Document QA project's real retrieval code (loaders, chunker, pgvector store).

The lab evaluates that project's retriever as-is rather than a copy. Both projects use a
top-level package named `src`, so Document QA is loaded under its own name, `docqa`.
Keep the two repositories side by side (or set DOCQA_PATH).
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


class DocqaNotFound(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_docqa(path: str) -> SimpleNamespace:
    root = Path(path).resolve() / "src"
    init = root / "__init__.py"
    if not init.exists():
        raise DocqaNotFound(
            f"Document QA project not found at {Path(path).resolve()}. Clone document-qa-citations next to "
            "this repository or set DOCQA_PATH.")
    if "docqa" not in sys.modules:
        spec = importlib.util.spec_from_file_location("docqa", init, submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["docqa"] = module
        spec.loader.exec_module(module)

    from docqa.ingest import chunk_sections, load_file  # noqa: E402
    from docqa.ingest.loaders import LoadedDocument, Section
    from docqa.tools.store import HYBRID_SQL, DocumentStore, keyword_query

    return SimpleNamespace(DocumentStore=DocumentStore, keyword_query=keyword_query, HYBRID_SQL=HYBRID_SQL,
                           chunk_sections=chunk_sections, load_file=load_file, LoadedDocument=LoadedDocument,
                           Section=Section, path=str(Path(path).resolve()))
