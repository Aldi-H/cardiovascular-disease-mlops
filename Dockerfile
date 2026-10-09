FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY serving/requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY serving/ /app/serving/

RUN addgroup --system appuser \
    && adduser --system --ingroup appuser --home /home/appuser appuser \
    && chown -R appuser:appuser /app /home/appuser

ENV HOME=/home/appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
