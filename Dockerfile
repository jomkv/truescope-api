FROM python:3.11-slim

# System packages needed for psycopg2, spaCy, torch builds
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN grep -v '^xx_ent_wiki_sm[[:space:]]*@' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt

# Download spaCy model into the image
RUN python -m spacy download xx_ent_wiki_sm

# Pre-download model assets into image cache (~/.cache/huggingface)
# This removes first-request download latency at runtime.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM; \
from sentence_transformers import SentenceTransformer; \
AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli'); \
AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli'); \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); \
AutoTokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws'); \
AutoModelForSeq2SeqLM.from_pretrained('Vamsi/T5_Paraphrase_Paws'); \
AutoTokenizer.from_pretrained('facebook/bart-large-cnn'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn'); \
print('All runtime models cached'); \
"

# Copy app code last (so code changes don't invalidate the model cache layers above)
COPY . .

# Ensure DO DB certificate exists at expected path used by DB_CA_PATH.
RUN test -f /app/certs/ca-certificate.crt

EXPOSE 8000

# Single worker only — multiple workers = each loads all models = OOM on 16GB
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]