FROM python:3.12-slim

# Reproducible, offline-friendly image. No network access is needed at runtime.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user to run the tool.
RUN useradd --create-home --uid 10001 runner

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir . && \
    chown -R runner:runner /app

USER runner

ENTRYPOINT ["knapmem"]
CMD ["bench", "--no-write"]
