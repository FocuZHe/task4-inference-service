# 任务 4：检测 ONNX

完成日期：2026-11-18。

## 做了什么

把 `yolo11n-dut-new` 和 `yolo11n-dut-old` 导出为 FP32、FP16、INT8 ONNX。推理输入按 Ultralytics letterbox 到 640，RGB，除以 255。类别 0 映射为 `drone`。用 6 张样例对比 PyTorch 与 ONNX，置信度阈值 0.25，NMS IoU 0.7。

INT8 使用 ONNX Runtime 动态量化。权重目录含中文时量化工具找不到中间文件，所以先复制到英文临时目录再转回。

## 文件

| 路径 | 作用 |
| --- | --- |
| `serving/detection_runtime.py` | 导出、ONNX 推理、与 PyTorch 对照 |
| `artifacts/onnx/yolo11n-dut-*-{fp32,fp16,int8}.onnx` | 六个检测 ONNX |
| `artifacts/reports/detection-parity.json` | 每个精度、每个样例的框数和误差 |
| `tests/test_detection_onnx.py` | FP32 类别一致率必须为 1 |

## ONNX SHA-256

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
| `yolo11n-dut-new-fp32.onnx` | 10,604,532 | `e5366b821dba6eb229ac9cdc97024b5f5ea52d18390231dc358682dacacf3351` |
| `yolo11n-dut-new-fp16.onnx` | 5,356,031 | `22500a0fd89d7ad9c60951106b6d6d390bf7bf25dffaf31b2f759b50c06eafe9` |
| `yolo11n-dut-new-int8.onnx` | 3,013,733 | `041a76eaf6cfb3bbac48dfb189a00b5c57b6c9bb4b8a9f10ff4d68be731a527c` |
| `yolo11n-dut-old-fp32.onnx` | 10,604,502 | `0e294b5bcc149d7db585f9f481fe4d3b8d717869952c28e9027419bfcbefc5a3` |
| `yolo11n-dut-old-fp16.onnx` | 5,356,001 | `30af80cf532dbd9a801873647b56fcdaaa7d78aa486a3550d042fde4bfcbfc2f` |
| `yolo11n-dut-old-int8.onnx` | 3,013,703 | `9c941a0548a803438cb6540d9ad20367d2a8f21e0c33573fd8fbfb633139e36c` |

## 对照结果

6 张样例、两个模型、三种精度的类别一致率都是 1.0。输出类别都是 `drone`。

FP32 最大中心距不超过 0.0003 像素，最大置信度绝对差不超过 0.000001。FP16 最大中心距 0.9113 像素，最大置信度绝对差 0.001401。INT8 类别仍一致，但定位误差更大：`yolo11n-dut-old` 在 `01769.jpg` 上框数 3 对 2，匹配中心距达到 1704.2619 像素，置信度绝对差 0.171474。这些数字只说明量化误差，不改变检测精度结论。

## 验证

```powershell
.\.venv-task4\Scripts\python.exe -m pytest tests/test_detection_onnx.py -v
```

结果：1 passed。
