# 任务四验收摘要

日期：2026-11-18。镜像 `task4-serving:latest` 已按稳定批量改动重新构建，清单 `sha256:347c1be77a2bc9b2d6b3367a6f8861e4fe8fcd078961b1466d6b6dd66a446441`。Docker 客户端和引擎都是 29.9.3。

## 工程

这些数字来自已完成的测试和报告，夹具没有拿来改写模型精度。

| 项目 | 结果 |
| --- | --- |
| 接口合同 | `tests/test_contracts.py`、`tests/test_api.py` 通过。响应含 `modelVersion`、`requestId`、`source`、`receivedAt`、`result`、`latencyMs`、`errorCode` |
| 检测 FP32 类别一致率 | 6 张样例、两个模型均为 1.0 |
| 画质 FP32 绝对误差 | 固定向量 `4.25e-8`；10 秒冒烟视频归一化画质分 `3.38e-8` |
| 单帧端到端 P50 / P95 / P99 | 107.13 / 140.5512 / 627.4662 ms。P99 有一次单帧尖峰，P95 仍低于 300 ms |
| 单帧 P95 ≤ 300 ms | 通过 |
| 8 帧批量吞吐 / 单帧吞吐 | 19.7499 / 8.1195 = 2.4324。另有 6 组 30 次测量，最低 2.409；单帧最快的一组是 8.947 帧/秒，比值仍是 2.805。复用连接和每次新开连接都过 2 倍 |
| 错误率 | 100 次延迟请求为 0 |
| 显存 | 0。没有 NVIDIA GPU |
| 六种故障 | `tests/test_faults.py` 6 passed。缺失、坏输入、超时、内存错误、队列满、重启都返回了明确错误码 |
| 切换和回滚 | `tests/test_registry.py` 中成功次数 / 尝试次数 = 1.0。画质成对切换。v13 未登记 |
| 8 小时稳定性 | 2880 次请求，28,804,346 ms，退出码 0，`unhandled` 为 0 |

批量吞吐按重复测量达标。8 个 FP32 会话各用 2 个线程，并关闭空转等待。JPEG/PNG 校验只读文件头。请求 JSON 用 orjson。原始数字在 `artifacts/reports/latency.json`。

画质 Ridge 另有一份 INT8 ONNX：`quality-ridge-v1-int8.onnx`，SHA-256 `75ed7a81d938d3570514438825abdaeb00051affcaeaf9e2c56225682b36e2a9`，314 字节。固定向量上 joblib 与 INT8 的绝对差是 `4.250342211520319e-08`。随机森林仍用 FP32，`TreeEnsembleClassifier` 属于 `ai.onnx.ml`，`quantize_dynamic` 不能量化。服务打分仍走 FP32。

镜像定义在 `docker/Dockerfile`。基础镜像 `python:3.11-slim`，安装 `libgl1` 和 OpenCV 需要的 `libglib2.0-0`，以非 root 用户 `appuser` 运行。环境变量 `TASK4_ORT_PROVIDER=CPUExecutionProvider`，检测和画质运行时都会读取它。启动命令是 `uvicorn serving.app:app --host 0.0.0.0 --port 8080`。

任务一、任务三的原始权重不放进镜像，运行时只读挂载：

```powershell
docker build -f docker/Dockerfile -t task4-serving .
docker run --rm -p 8080:8080 `
  -v ${PWD}/任务一模型:/app/任务一模型:ro `
  -v ${PWD}/任务三模型:/app/任务三模型:ro `
  task4-serving
```

构建已完成。CPU 版 PyTorch 使用镜像上下文里的 `docker/wheels/torch-2.15.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`，其余依赖走清华 PyPI。挂载任务一、任务三权重后，容器以 `uid=10001(appuser)` 启动，`GET /health` 和 `GET /ready` 都返回 `{"status":"ok"}`。验证容器随后已删除。

## 模型

下面是任务一、任务三交付时已经记录的指标。本次夹具、ONNX 对照和 8 小时运行都没有重算这些数字。

| 来源 | 已有指标 |
| --- | --- |
| 任务一，清理版 val 2599 张 | mAP50 由 35.22% 到 85.72%，mAP50–95 由 15.45% 到 53.91% |
| 任务三交付说明，KoNViD-1k 官方 test | 质量回归 SRCC `0.6215044650236221` |
| 任务三交付说明，LIVE-VQC external_test | 视频级 SRCC `0.5445168098704705`。该数据未参与训练 |
| 任务三交付说明，受控异常 candidate-only 正式 test | Macro-F1 `0.919403`。这不是默认 baseline，也不能当成现场验收 |

检测 ONNX 的类别一致和画质 ONNX 的分数差只说明导出后数值是否保持，不表示识别精度或画质相关指标提高。
