FROM ghcr.io/astral-sh/uv:0.9.24 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY compose.yaml Dockerfile ./
RUN uv sync --locked --no-dev
COPY src ./src
COPY data ./data
COPY sql ./sql
ENTRYPOINT ["/app/.venv/bin/python", "-m", "src.cli"]
CMD ["--help"]
