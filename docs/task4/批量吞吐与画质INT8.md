# 批量吞吐与画质 INT8

完成日期：2026-11-18。

## 做了什么

8 帧检测使用 8 个 FP32 会话，每个会话 2 个线程，并关闭空转等待。线程池在进程里复用。JPEG 和 PNG 的长边检查只解码文件头。请求体用 orjson 解析。单帧接口仍用占满 CPU 的那一个 FP32 会话。

画质 Ridge 写成 6 个 int8 权重加 `DequantizeLinear` 和 `Gemm`，保存为 `quality-ridge-v1-int8.onnx`。标准化和缺失值填充折进这条直线。服务打分继续用 FP32。随机森林不生成 INT8 文件，报告写明 `TreeEnsembleClassifier` 在 `ai.onnx.ml`，`quantize_dynamic` 不能量化。

## 文件

| 路径 | 作用 |
| --- | --- |
| `serving/detection_runtime.py` | 8 路并行 FP32 推理 |
| `serving/inference.py` | 批量接口走编码后的并行入口 |
| `serving/contracts.py` | JPEG/PNG 文件头尺寸 |
| `serving/quality_runtime.py` | Ridge INT8 和森林不可量化说明 |
| `benchmark/latency.py` | 复用 HTTP 连接并重写报告 |
| `artifacts/reports/latency.json` | 100 次重测 |
| `artifacts/reports/quality-parity.json` | Ridge INT8 误差和森林说明 |
| `artifacts/onnx/quality-ridge-v1-int8.onnx` | 314 字节 |

## 验证

| 命令 | 结果 |
| --- | --- |
| `python -m pytest tests/test_contracts.py tests/test_api.py tests/test_faults.py tests/test_detection_batch.py tests/test_quality_int8.py -q` | 32 passed |
| `python -m benchmark.latency --requests 100` | 退出码 0，错误率 0，比值 2.4324，单帧 P95 140.5512 ms |
| 6 组 30 次，含复用连接和每次新开连接 | 比值 2.409、2.520、2.923、3.098、2.805、3.069 |

## 关键数字

- 单帧 P50 / P95 / P99：131.801 / 159.5366 / 201.5154 ms
- 单帧吞吐：7.5438 帧/秒
- 8 帧吞吐：15.5569 帧/秒
- 批量 / 单帧：100 次正式结果 2.4324。重复测量最低 2.409。单帧最快 8.947 帧/秒时比值 2.805
- Ridge INT8 SHA-256：`75ed7a81d938d3570514438825abdaeb00051affcaeaf9e2c56225682b36e2a9`
- 固定向量绝对差：`4.250342211520319e-08`
