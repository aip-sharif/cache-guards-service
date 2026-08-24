FROM docker.abrha.net/python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# system deps needed to build psycopg2 (if not using -binary) and for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock.txt

COPY . .

# [R03 FIX] اجرا با یوزر غیرروت - قبلاً همه‌چیز به‌عنوان root اجرا می‌شد
# (یعنی اگه یه vulnerability توی کد/dependency پیدا بشه، مهاجم مستقیم
# root داخل کانتینر می‌شه)
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown -R appuser:appuser /code

USER appuser

EXPOSE 8000

# [R03 FIX] HEALTHCHECK قبلاً اصلاً نبود - یعنی Docker/orchestrator هیچ‌وقت
# نمی‌فهمید سرویس واقعاً سالمه یا فقط پروسه‌ش بالاست ولی جواب نمی‌ده
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
