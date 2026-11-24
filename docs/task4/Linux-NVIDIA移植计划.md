# 任务四 Linux／NVIDIA 移植计划

## 部署方案

首轮按 Linux x86_64、Docker 和单张已分配的 NVIDIA GPU 准备。GPU 镜像使用 Python 3.11、ONNX Runtime GPU 1.26.0、CUDA 12.8 Update 1 和 cuDNN 9。Linux 驱动按 570.124.06 或以上作为首选兼容基线，仍需核对目标 GPU 型号与驱动分支，并验证容器内的实际推理。CUDA 用户态库随镜像安装。 [ONNX Runtime CUDA 兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) · [NVIDIA CUDA 12.8 Update 1 驱动矩阵](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html) · [NVIDIA Container Toolkit 验证](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)

检测用 CUDA FP32，关闭 TF32。应用启动时预热活动模型的单帧会话，并在存在批量模型时预热 8 帧会话。CUDA 初始化失败会终止启动。ONNX Runtime 可为 CUDA 未覆盖的算子分配 CPU 执行。活动检测模型使用经过验证的 `yolo11n-dut-new-batch-fp32.onnx` 一次处理 8 帧，并按输入顺序返回。其他批量大小和缺少批量模型的版本沿用逐帧 CUDA 推理。画质在 CPU 上运行。验收用固定夹具核对输出和性能，原交付的 Windows CPU 指标作为历史记录。

## 服务器预检和传输

在服务器上先确认系统和硬件，记录输出用于验收。

```bash
uname -m
cat /etc/os-release
docker version
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used --format=csv
```

按目标盘的实际可用空间和服务器网络准备镜像及模型。联网环境可在服务器构建。离线环境可先在可联网的 Linux x86_64 构建机上构建，用 `docker save` 和 `docker load` 转移镜像。保存镜像 ID、依赖版本及传输件的 SHA-256。

| 传输内容 | 要求 |
| --- | --- |
| 构建目录 | `docker/Dockerfile.gpu`、`docker/Dockerfile.gpu.dockerignore`、`requirements-task4-gpu.txt`、`serving/`、`benchmark/` 与 Dockerfile 引用的注册表、FP32 ONNX、1080p 夹具 |
| 运行时模型 | 保留 `artifacts/registry/models.json` 列出的所有相对路径，包括回滚权重。两个模型目录在容器中只读挂载。Linux 上目录名、文件名和大小写须一致 |
| 额外验证样例 | 复用 `任务一模型/demo/samples/` 的样例和 `任务三模型/data/raw/task3-smoke/engineering-smoke-10s.avi` |

`ModelRegistry.load()` 会核验所有登记的源权重 SHA-256。另对 FP32 ONNX 和固定夹具在传输前后核对 SHA-256。容器用户 UID 为 10001，须能读取模型文件并遍历其父目录。

## 构建和隔离验证

在完整工程目录运行。先由服务器负责人确定允许使用的 GPU UUID，并设置 `TASK4_GPU_UUID`。CPU 对照和 CUDA 候选使用同一份镜像、模型和数据。

```bash
: "${TASK4_GPU_UUID:?请先设置本次分配的 GPU UUID}"
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t task4-serving:linux-gpu-stage .
IMAGE=task4-serving:linux-gpu-stage
docker run --rm --gpus "device=$TASK4_GPU_UUID" --entrypoint nvidia-smi "$IMAGE" -L
docker image inspect --format '{{.Id}}' "$IMAGE"

model_mounts=(
  --mount "type=bind,source=$PWD/任务一模型,target=/app/任务一模型,readonly"
  --mount "type=bind,source=$PWD/任务三模型,target=/app/任务三模型,readonly"
)
docker run -d --name task4-linux-cpu -p 127.0.0.1:18080:8080 \
  -e TASK4_ORT_PROVIDER=CPUExecutionProvider "${model_mounts[@]}" "$IMAGE"
docker run -d --name task4-linux-cuda -p 127.0.0.1:18081:8080 \
  --gpus "device=$TASK4_GPU_UUID" "${model_mounts[@]}" "$IMAGE"

docker logs task4-linux-cuda
for port in 18080 18081; do
  curl --fail --silent --show-error "http://127.0.0.1:$port/health"
  curl --fail --silent --show-error "http://127.0.0.1:$port/ready"
done
curl --fail --silent --show-error http://127.0.0.1:18081/metrics
```

CUDA 启动日志应出现 `warmed up with CUDAExecutionProvider`。运行压测时同时观察所分配 GPU 的占用，以及是否出现驱动或 ORT 错误。 `gpu_memory_bytes` 是容器可见 GPU 的设备级总已用显存，在共享 GPU 上包含同卡其他进程。采集失败显示 `NaN`。

用报告中的六张已有样例，在两个模型版本上对比 CPU 和 CUDA 的单帧输出。逐图检查框数量和类别，按中心距离匹配并记录位置及置信度差。验收参考此前的 5 像素中心距离和 0.05 置信度差阈值。另测 8 帧结果数量和顺序、固定视频、画质接口、坏输入及重启。用同版本镜像尝试在无 GPU 分配的容器中以 CUDA 模式启动，预期启动失败。

## 性能和切换

避开其他负载高峰，按顺序运行 CPU 与 CUDA benchmark。记录 GPU 型号、驱动、镜像 ID 和共享负载。测试用的请求超时应与拟上线配置一致。

```bash
report_dir="linux-validation/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$report_dir"
for backend in cpu cuda; do
  container="task4-linux-$backend"
  report="latency-$backend-${report_dir##*/}.json"
  docker exec "$container" python -m benchmark.latency --existing-server --port 8080 \
    --requests 100 --output "/tmp/$report" || printf '%s\n' "$backend benchmark reported an error"
  docker cp "$container:/tmp/$report" "$report_dir/" || docker logs "$container"
done
docker exec task4-linux-cuda python -m benchmark.soak --existing-server --port 8080 \
  --duration-s 30 --output /tmp/soak-cuda-smoke.jsonl
docker cp task4-linux-cuda:/tmp/soak-cuda-smoke.jsonl "$report_dir/"
```

检查报告中的 P50、P95、P99、吞吐、错误率、RSS、显存和 8 帧与单帧的吞吐比。历史门槛是单路 1080p P95 ≤ 300 ms、8 帧吞吐比 ≥ 2×、8 小时无未处理异常退出。候选容器先通过短测，再以 `--duration-s 28800` 执行 8 小时测试并保存独立报告，同时检查容器重启次数和 GPU 错误。若 GPU 批量吞吐比未达门槛，保留实际测量并单独评估后续批量实现。

验收通过后，先通过现有入口对少量调用方切换到 CUDA 候选，并观察错误率、延迟和显存。出现启动、数值对照或稳定性问题时，将流量切回已验证的 Linux CPU 实例，并保留 CUDA 的日志与报告便于排查。CPU 和 CUDA 镜像、模型版本及入口切换点在正式切换前记录。

## 平台日志

日志工作在服务验证通过后开展。先在服务器上用只读权限选取一小段已有日志，确认格式、时间戳和时区、时间跨度、写入或轮转方式、请求 ID 的可用性，以及是否关联可获授权的媒体。以脱敏样例确定字段映射。

首轮用离线汇总检查时间分布、缺失及重复记录，并在请求 ID 可关联时与服务响应中的延迟和错误码做对照。访问日志与聚合指标用于同时间窗排查。只有日志关联的媒体才能用于推理回放，回放使用隔离实例和小样本。输出字段映射、样例覆盖率、解析失败和对照结果。原日志留在服务器，以只读方式访问，分析产物单独存放。
