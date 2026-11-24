# 任务 10：Dockerfile 与验收摘要

完成日期：2026-11-18。镜像已构建并做过启动检查。

## 做了什么

补上镜像构建。容器内安装 CPU 版 PyTorch 2.15.0 时，在线下载两次都在中途截断，所以把完整 wheel 放进 `docker/wheels/`，构建时按原文件名安装。其余依赖使用清华 PyPI，Debian 软件包使用清华镜像。检测和画质的 ONNX Runtime 提供方读取 `TASK4_ORT_PROVIDER`，镜像里设为 `CPUExecutionProvider`。

验收摘要仍把工程指标和任务一、任务三已有精度分开。8 帧吞吐的 100 次正式结果是 2.4324，重复测量最低 2.409。

## 文件

| 路径 | 作用 |
| --- | --- |
| `docker/Dockerfile` | `python:3.11-slim`，非 root，端口 8080 |
| `docker/wheels/torch-2.15.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl` | Linux CPU 版 PyTorch，避免构建时下载被截断 |
| `.dockerignore` | 排除虚拟环境和缓存 |
| `artifacts/reports/task4-acceptance.md` | 工程指标和原有模型指标分开，并写上镜像结果 |

任务一、任务三的原始权重通过只读挂载进入容器，不复制进镜像。

## 验证

| 命令 | 结果 |
| --- | --- |
| `docker version` | 客户端 29.9.3，引擎 29.9.3 |
| `docker build -f docker/Dockerfile -t task4-serving .` | 稳定批量改动后重构建，退出码 0。标签 `task4-serving:latest`，清单 `sha256:347c1be77a2bc9b2d6b3367a6f8861e4fe8fcd078961b1466d6b6dd66a446441` |
| 挂载权重后 `GET /health`、`GET /ready` | 都是 `{"status":"ok"}`。进程用户 `uid=10001(appuser)`。检查后容器已删除 |
| `python -m pytest tests -q --ignore=tests/test_quality_onnx.py` | 34 passed，44.18 秒 |
| `tests/test_quality_onnx.py` | 此前单独跑过，3 passed，1344.94 秒。这次没有重跑 |

## 关键数字

- 镜像清单：`sha256:347c1be77a2bc9b2d6b3367a6f8861e4fe8fcd078961b1466d6b6dd66a446441`
- 镜像大小：5.48GB
- 单帧 P95：140.5512 ms，通过 300 ms 上限
- 8 帧吞吐 / 单帧吞吐：2.4324。重复测量最低 2.409
- 8 小时稳定性：2880 次请求，28,804,346 ms，退出码 0
