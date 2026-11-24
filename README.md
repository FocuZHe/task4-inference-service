# 任务四算法服务交付

任务四的模型推理服务部署在内网服务器 `192.168.153.223`，交付目录为 `/srv/task4`。检测服务使用 0 号 NVIDIA RTX 3090 和 CUDA FP32，画质服务使用 CPU。部署日期为 2026 年 11 月 24 日。

## 访问服务

- API 基础地址为 `http://192.168.153.223:18781`。可直接打开[健康检查](http://192.168.153.223:18781/health)。
- 接口文档在 [Swagger UI](http://192.168.153.223:18781/docs)，机器可读的接口定义在 [OpenAPI JSON](http://192.168.153.223:18781/openapi.json)。
- 从可访问该服务器的内网环境执行以下命令。两个接口均应返回 `{"status":"ok"}`。

```bash
curl -fsS http://192.168.153.223:18781/health
curl -fsS http://192.168.153.223:18781/ready
```

在服务器上可以用随包样例验证单帧、8 帧合批和画质视频接口。

```bash
python3 /srv/task4/bin/smoke.py http://192.168.153.223:18781
```

| 接口 | 输入 |
| --- | --- |
| `POST /infer/frame` | 单张图片的 `imageBase64` |
| `POST /infer/batch` | 1 至 8 张图片的 `framesBase64` 数组 |
| `POST /infer/video` | 短视频的 `videoBase64` 或已登记的 `fixtureId` |
| `POST /infer/video?task=quality` | 视频或夹具，模型版本用 `quality-ridge-v1` |

每个请求填写 `requestId`、`modelVersion`、`source` 和带时区的 `receivedAt`。检测默认推荐 `yolo11n-dut-new`，验收样例的 `source` 为 `eo`。批量接口收到 8 帧时使用一次 CUDA 合批推理。接口定义提供完整字段和约束。

## 服务器文件

| 路径（相对于 `/srv/task4`） | 内容 |
| --- | --- |
| `release/serving/`、`release/benchmark/`、`release/tests/` | 服务、压测和回归代码 |
| `release/artifacts/onnx/` | 检测、画质和事件模型的 ONNX 文件，含部署使用的 8 帧 FP32 模型 |
| `release/任务一模型/`、`release/任务三模型/` | 登记的原始权重和测试样例 |
| `release/artifacts/registry/models.json` | 模型版本、源权重 SHA-256 和回滚关系 |
| `release/artifacts/fixtures/` | 固定测试视频和生成脚本 |
| `release/artifacts/reports/linux-gpu-acceptance-2026-11-23/` | 最终验收报告、8 小时逐请求记录、GPU 与健康采样、回归和对照结果 |
| `release/artifacts/reports/` | 历史 CPU 验收、GPU 试跑和正式切换记录 |
| `release/SHA256SUMS` | 已上传交付文件的校验清单 |
| `runtime/`、`bin/` | 服务器运行依赖、容器创建脚本和冒烟脚本 |
| `verification.json` | 发布内网端口后的服务验证结果 |

在服务器上执行 `cd /srv/task4/release && sha256sum --strict -c SHA256SUMS` 可核对交付文件。

## 验收结果

任务书的 7 项指标全部通过。以下数据来自同一发布代码和模型在服务器隔离验证时的记录。

- 接口合同与 API 测试通过 **24/24**，完整回归通过 **48/48**。
- PyTorch 与 ONNX 的 20 组固定输入对照中，检测类别和框数一致。画质固定向量的绝对误差为 `4.25e-8`。
- 单帧 1080p 端到端 **P95 为 64.3 ms**，门槛为 300 ms。
- 8 帧批量吞吐为 **36.87 帧/秒**，是单帧 **16.31 帧/秒**的 **2.26 倍**。
- 11 月 23 日 21:02 至 11 月 24 日 05:02（香港时间）的 8 小时稳测中，**2,880 次请求全部成功**，未处理异常和容器重启均为 0。
- 故障注入通过 **6/6**。模型切换与回滚在临时注册表上通过 **8/8**。

发布内网入口后，服务器完成了单帧、8 帧合批和画质视频冒烟验证。内网工作站也通过了健康检查、就绪检查、接口文档访问和单帧推理。

## 运维

容器名为 `task4-serving`，使用宿主机网络并监听 `192.168.153.223:18781`。容器只使用 0 号 GPU，配额为 8 个逻辑 CPU、4 GiB 内存，模型与发布目录以只读方式挂载。重启策略为 `unless-stopped`。

```bash
sudo docker inspect --format '{{.State.Status}} {{.State.Health.Status}}' task4-serving
sudo docker logs --tail 100 task4-serving
```

需要重新启动时，先确认 0 号 GPU 资源可用。已有容器用 `sudo docker start task4-serving` 启动；容器删除后，可用 `sudo /srv/task4/bin/create-container.sh` 重建，启动并通过冒烟后用 `sudo docker update --restart unless-stopped task4-serving` 恢复自动重启策略。
