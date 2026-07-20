FROM python:3.12-slim AS builder

WORKDIR /app

ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV HF_HOME=/app/.cache/huggingface

# Install uv package manager
RUN pip install uv

# Copy dependency configuration
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (creates .venv)
RUN UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --no-dev

# Keep inference startup deterministic and independent of runtime network
# access by baking the configured sentence-transformers model into the image.
RUN .venv/bin/python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])"

# Copy the rest of the application code
COPY . .

# Runner stage
FROM python:3.12-slim AS runner

WORKDIR /app

ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV HOME=/home/gh-social
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy the environment and app from the builder
COPY --from=builder /app /app

# Create a non-root user with writable cache locations for model/runtime state.
RUN addgroup --system gh-social \
    && adduser --system --ingroup gh-social --home /home/gh-social gh-social \
    && mkdir -p /home/gh-social /app/.cache \
    && chown -R gh-social:gh-social /home/gh-social /app/.cache
USER gh-social

# Update PATH to prioritize virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

# Start the ML API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
