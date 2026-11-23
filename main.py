"""Entry point.

    uvicorn main:app --reload        # API (docs at /docs)
    python main.py                   # same, on port 8000
"""

from __future__ import annotations

from src.api import create_app
from src.utils import load_settings, setup_logging

settings = load_settings()
setup_logging(settings.log_level, settings.log_dir)
app = create_app(settings)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
