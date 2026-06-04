FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIM_APP_HOST=0.0.0.0 \
    SIM_APP_PORT=8050

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py new_sim_app.py bayes_classifier_app.py process_eda.py ./
COPY assets ./assets

RUN mkdir -p /app/outputs/uploads /app/data

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${SIM_APP_PORT}/_dash-layout >/dev/null || exit 1

CMD ["python", "-u", "new_sim_app.py"]
