# syntax=docker/dockerfile:1

FROM python:3.10.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ecoface_lite ./ecoface_lite
COPY dashboard ./dashboard

EXPOSE 8000

# Default: API. Override CMD to run Streamlit, e.g.:
# streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "ecoface_lite.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
