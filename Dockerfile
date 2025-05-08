# ---- 构建阶段 ----
FROM python:3.11-slim AS builder
WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y build-essential gcc libglib2.0-0 libsm6 libxext6 libxrender-dev libmagic1 pandoc libreoffice && rm -rf /var/lib/apt/lists/*

# 拷贝依赖文件并安装
COPY requirements-docker.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

# ---- 运行阶段 ----
FROM python:3.11-slim
WORKDIR /app

# 拷贝依赖
COPY --from=builder /install /usr/local

# 拷贝项目代码和静态文件
COPY . .

# 环境变量（可按需修改）
ENV PYTHONUNBUFFERED=1 \
    QA_PASSWORD=changeme \
    QA_API_TOKEN=changeme

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 