# ONNX 与任务二/任务四状态

## 当前事实

- 本包没有 `.onnx` 文件。
- 当前四份模型是 scikit-learn/joblib artifact：两份正式 baseline、两份 `candidate-v2` 灰度/回滚候选。
- 当前服务使用 `joblib.load`，再做模型版本、特征列、candidate opt-in 和 SHA-256 校验；它不是 ONNX Runtime 服务。
- 没有完成 ONNX 导出、CPU/GPU provider 对比、FP32/FP16/INT8 量化、固定输入误差、批量吞吐或 HTTP P95 证据。

## 任务二/任务四还要做

1. 在不改动四份原始 `.joblib` 的前提下，建立独立 `onnx/` 版本和转换脚本。
2. 对每个转换版本登记输入特征列、模型版本、转换器版本、SHA-256 和 golden 输出。
3. 用同一批固定数值输入比较 joblib 与 ONNX 输出；记录最大绝对误差、平均绝对误差和分类/回归结论，不用测试集重新选模型。
4. 测量 ONNX Runtime CPU；有 NVIDIA/CUDA 条件时再测 CUDA/TensorRT provider。没有 provider 时明确回退 CPU，不把“机器有GPU”写成“服务用了GPU”。
5. 另测 FP32、FP16、INT8 的精度差异、P50/P95/P99、吞吐、队列等待、内存和显存；保存原始报告。
6. 加入模型缺失、SHA不匹配、输入损坏、超时、队列满、provider不可用、进程重启和回滚测试。

在这些证据完成之前，任务二/任务四的 ONNX 服务状态应写为 `not_ready`，不能通过创建空文件、改扩展名或使用随机转换结果来满足前置条件。

