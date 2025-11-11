FROM python:3.10-slim

WORKDIR /app

# Java for Spark
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY data/ ./data

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
