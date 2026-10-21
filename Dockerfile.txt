FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1 ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Pre-download Kokoro model + voices at build time (avoids slow cold starts)
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a'); KPipeline(lang_code='b')"

COPY app.py .

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 180
