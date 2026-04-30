FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY deploy/ ./deploy/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY .env ./

# Install dependencies
RUN uv sync --frozen

# Copy and set entrypoint
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
