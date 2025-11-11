cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y default-jdk curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/jars && \
    curl -L -o /app/jars/postgresql-42.7.3.jar https://jdbc.postgresql.org/download/postgresql-42.7.3.jar

COPY ../src ./src
COPY ../data ./data

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
EOF



