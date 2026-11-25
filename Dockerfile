# Optional Stage 11 extension. Requires trained artifacts in ./artifacts
# (run the training commands from the README first; *.joblib is git-ignored).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6"

COPY src ./src
COPY api ./api
COPY configs ./configs
COPY artifacts ./artifacts

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
