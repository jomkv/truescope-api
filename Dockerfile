FROM python:3.11-slim

# System packages needed for psycopg2, spaCy, torch builds
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN python - <<'PY'
from pathlib import Path

src = Path("requirements.txt")
raw = src.read_bytes()

text = None
for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
    try:
        text = raw.decode(enc)
        break
    except Exception:
        pass

if text is None:
    raise SystemExit("Could not decode requirements.txt")

out_lines = []
for line in text.splitlines():
    stripped = line.strip()
    if stripped.lower().startswith("xx_ent_wiki_sm @"):
        continue
    out_lines.append(line)

Path("/tmp/requirements.docker.txt").write_text(
    "\n".join(out_lines) + "\n",
    encoding="utf-8",
)

print(f"Prepared /tmp/requirements.docker.txt with {len(out_lines)} lines")
PY
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

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
print('All runtime models cached'); \
"

# Copy app code last (so code changes don't invalidate the model cache layers above)
COPY . .

# Ensure DO DB certificate exists at expected path used by DB_CA_PATH.
RUN test -f /app/certs/ca-certificate.crt

# Runtime tuning knobs for verification pipeline (override at deploy time if needed)
ENV NLI_TORCH_NUM_THREADS=1 \
    VERIFY_NLI_MAX_THREADS=2 \
    VERIFY_MAX_THREADS=3 \
    VERIFY_MAX_CONCURRENT_PROCESSES=5 \
    VERIFY_PER_REQUEST_CONCURRENCY_LIMIT=2 \
    VERIFY_DB_RETRIEVE_LIMIT=20 \ 
    VERIFY_AGGREGATION_LIMIT=3 

EXPOSE 8000

# Start with 2 workers to better utilize available RAM and improve throughput.
# Each worker loads model state, so increase cautiously beyond 2.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info", "--ws", "websockets-sansio"]