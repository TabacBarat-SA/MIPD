FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY cliente360 ./cliente360
COPY services ./services
COPY --from=web /web/dist ./web/dist
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CMD ["python", "services/run_gateway.py"]
