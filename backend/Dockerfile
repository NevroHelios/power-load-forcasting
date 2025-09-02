FROM astral/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# COPY requirements.txt ./
COPY pyproject.toml uv.lock ./

# RUN pip install -r requirements.txt
RUN uv sync

COPY . .

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
