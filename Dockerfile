FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md CLAUDE.md ./
COPY core ./core
COPY agents ./agents
COPY social_memory ./social_memory
COPY moltbook ./moltbook
COPY config.py runner.py SOUL.md ./

# Install core deps plus PostgreSQL support (psycopg2-binary extra).
RUN pip install --upgrade pip && pip install ".[postgres]"

CMD ["python", "runner.py"]
