# 任务 5：画质 ONNX

完成日期：2026-11-18。

## 做了什么

用 skl2onnx 把 baseline 画质 Ridge pipeline 和异常随机森林转成 opset 12 的 FP32 ONNX，输入名是 `features`。固定向量和 10 秒冒烟视频都用 joblib 对照。FP16 两份都转出来了。`quantize_dynamic` 不能量化 `ai.onnx.ml`。Ridge 后来另存了一份按特征量化的 INT8 图，随机森林继续用 FP32。服务打分仍走 FP32。

## 文件

| 路径 | 作用 |
| --- | --- |
| `serving/quality_runtime.py` | 转换、`score_features`、`event_proba`、冒烟对照 |
| `artifacts/onnx/quality-ridge-v1-fp32.onnx` | Ridge FP32 |
| `artifacts/onnx/quality-ridge-v1-fp16.onnx` | Ridge FP16 |
| `artifacts/onnx/event-rf-v1-fp32.onnx` | 随机森林 FP32 |
| `artifacts/onnx/event-rf-v1-fp16.onnx` | 随机森林 FP16 |
| `artifacts/reports/quality-parity.json` | 分数差、事件类别、冒烟区间、转换错误 |
| `tests/test_quality_onnx.py` | 固定向量、类别概率、冒烟视频 |

特征顺序保持任务三的定义。画质是 `lap_var_mean`、`luma_mean`、`dark_ratio_mean`、`bright_ratio_mean`、`contrast_mean`、`frame_diff_mean`。异常是 `luma_mean`、`luma_std`、`dark_ratio`、`bright_ratio`、`lap_var`、`frame_diff`。冒烟视频特征由 `scripts/extract_combined_features.py` 提取。

## 哈希

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
| `quality-ridge-v1-fp32.onnx` | 572 | `84183a4e7d0618652cf60a1ed584703a408d1fe5c79ed5d526abadf9b4f5931c` |
| `quality-ridge-v1-fp16.onnx` | 1,152 | `ed607afeb108b05be950a192c0708a893fac31bc0c3858faf7866e1b2a40ce19` |
| `event-rf-v1-fp32.onnx` | 354,682,586 | `1f08f616979f36f76f37eab1244b4718fff3a26f705f1a40245a863bb0d457b7` |
| `event-rf-v1-fp16.onnx` | 354,683,166 | `f3dc6c151dabd25d571b4c827864314c89c45263a5b93f02484d9455018f6e39` |

## 对照

固定向量 `[120, 110, 0.02, 0.01, 40, 3.5]` 的 Ridge 原始输出差是 `4.25e-8`。seed 42 的异常向量两边类别都是 `normal`，概率差小于 `1e-4`。

冒烟视频 `engineering-smoke-10s` 的归一化画质分差是 `3.38e-8`。joblib 事件区间只有 `overexposed`，从 0.48 秒到 10.0 秒。报告没有 Macro-F1。

INT8 转换错误：`ValueError: Failed to find proper ai.onnx domain`。FP32 仍是服务用的版本。

## 验证

```powershell
.\.venv-task4\Scripts\python.exe -m pytest tests/test_quality_onnx.py -v
```

结果：3 passed。随机森林 ONNX 首次加载约 22 分钟，之后复用同一个运行时。
