# 合同文件

接口说明在仓库根目录的 [README.md](../README.md)。

| 路径 | 内容 |
| --- | --- |
| [`../shared_contracts`](../shared_contracts) | Pydantic 模型 |
| [`examples`](examples) | 合法 JSON 样例 |
| [`schemas`](schemas) | 由模型导出的 JSON Schema |
| [`tests/test_contracts.py`](tests/test_contracts.py) | 样例和约束测试 |
| [`requirements.txt`](requirements.txt) | `pydantic>=2.6` |

在仓库根目录执行：

```bash
python -m unittest contracts.tests.test_contracts
```
