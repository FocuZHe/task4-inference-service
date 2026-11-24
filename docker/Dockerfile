FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TASK4_ORT_PROVIDER=CPUExecutionProvider

RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-task4.txt .
COPY docker/wheels/torch-2.15.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl /tmp/torch-2.15.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl
RUN grep -v -E '^torch([<>= ].*)?$' requirements-task4.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir /tmp/torch-2.15.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl \
    && rm /tmp/torch-2.15.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements-no-torch.txt

COPY serving ./serving
COPY artifacts/registry ./artifacts/registry
COPY artifacts/onnx ./artifacts/onnx
COPY artifacts/fixtures ./artifacts/fixtures

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p \
        /app/任务一模型/demo/models \
        /app/任务三模型/data/raw/konvid-1k/models-official \
        /app/任务三模型/data/raw/konvid-1k/synthetic/full-frame-aligned-v3/models \
        /app/任务三模型/data/raw/konvid-1k/candidate-v2/events \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
