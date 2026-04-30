#!/bin/bash
set -e

# Run Alembic migrations
echo "Running database migrations..."
uv run alembic upgrade head

# Execute the passed command
exec "$@"
