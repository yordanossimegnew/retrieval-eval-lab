# Build from the PARENT folder so the Document QA project under test is included:
#   docker build -f retrieval-eval-lab/Dockerfile .
FROM python:3.12-slim

WORKDIR /app
COPY retrieval-eval-lab/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY document-qa-citations /document-qa-citations
COPY retrieval-eval-lab .
ENV DOCQA_PATH=/document-qa-citations
RUN useradd --create-home app && mkdir -p /app/data/benchmarks && chown -R app:app /app/logs /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
