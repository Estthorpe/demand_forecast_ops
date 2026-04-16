#!/bin/bash
set -e

echo "Configuring DVC credentials..."
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "${DAGSHUB_USERNAME}"
dvc remote modify origin --local password "${DAGSHUB_TOKEN}"

echo "Pulling model artefacts..."
dvc pull

echo "Starting server..."
exec uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 1
