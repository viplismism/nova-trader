FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

# Bind to the port the platform injects (Azure Container Apps / Cloud Run set $PORT);
# falls back to 8000 locally. Shell form so $PORT expands at runtime.
CMD ["sh", "-c", "nova web --host 0.0.0.0 --port ${PORT:-8000}"]
