cat << 'EOF' > Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y default-jdk curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 你已经把 Python 里的 spark.jars 改成 /app/jars/postgresql-42.7.3.jar
# 所以这里保持一致，放到 /app/jars 下面
RUN mkdir -p /app/jars && \
    curl -L -o /app/jars/postgresql-42.7.3.jar https://jdbc.postgresql.org/download/postgresql-42.7.3.jar

COPY src ./src
COPY data ./data

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
EOF






