FROM python:3.11-slim

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" '.[api]'

RUN adduser --disabled-password --gecos '' --uid 10001 pdserve
USER pdserve

EXPOSE 8200
CMD ["pdserve", "serve", "--host", "0.0.0.0", "--port", "8200"]
