#!/usr/bin/env bash
#
# One-command launcher for the AI Service.
# Activates the virtual environment and starts the FastAPI server.
#
# Usage:
#   ./run.sh            # start server on port 8000
#   ./run.sh 9000       # start server on a custom port
#
set -e

# Directory of this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PORT="${1:-8000}"

# Use the local venv if it exists
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  # Make sure all declared dependencies are installed (idempotent).
  pip install -q -r requirements.txt
fi

echo "=============================================="
echo "  Starting AI Service on http://localhost:${PORT}"
echo "  Swagger UI: http://localhost:${PORT}/docs"
echo "=============================================="

exec uvicorn app.main:app --reload --port "${PORT}"
