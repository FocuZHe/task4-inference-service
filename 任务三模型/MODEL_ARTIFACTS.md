# 模型登记与混用规则

本包内四份 `.joblib` 已随包提供。运行前必须按 `MODEL_HASHES.json` 或 README 中的 SHA-256 逐一核验；哈希不一致不得加载。

## 正式 baseline

- 质量：`data/raw/konvid-1k/models-official/quality_ridge.joblib`，`ridge-v1`，6维质量特征，SHA-256 `1cbb8558e01aecaa57d01a80cc578618fdf61ada2d0ac1164fe714aa4f87e535`。
- 异常：`data/raw/konvid-1k/synthetic/full-frame-aligned-v3/models/event-rf.joblib`，`task3-event-rf-v1`，类别 `blur/dark/overexposed/freeze_suspected`，SHA-256 `7ff590c07a66042ffd42233391a213aefdd37d8e01446258fe147412658a9e1e`。

## candidate-v2 灰度/回滚

- 质量：`data/raw/konvid-1k/candidate-v2/quality_candidate.joblib`，`hist_iter_100_leaf_7_l2_1-candidate-v2`，SHA-256 `0b9e3316d63d6a88fb3cedebee945c7611fef7635aa6d84259c24687e9e6996f`。
- 异常：`data/raw/konvid-1k/candidate-v2/events/event-v2-candidate.joblib`，`hgb_160_leaf_7_lr_0.05-candidate-v2`，SHA-256 `72304ed01a07520bb5c281c466c5e430dfb8389d30c92a0bebd4d029e0211363`。

上述两个 candidate-v2 只有在 `TASK3_MODEL_PROFILE=candidate` 且两份显式 SHA-256 都匹配时才能加载。candidate-v2 与 baseline 不能跨文件组合；质量和异常必须来自同一部署 profile 的登记组合。

## 不放入 candidate-v2 的 v13 候选

`data/raw/konvid-1k/synthetic/experiments/formal-candidate-v13-2026-11-12/candidate/event-v2-candidate.joblib`，SHA-256 `f5053c500443490b76c13c5ce1b23cce6867545fa11385c2888d5bc656995d22`，版本为 `rf_160_leaf_2_features_sqrt__blur_weight_4__confirm-3-recover-1.0-min-1.0-candidate-v2`，是 `event-features-v13` 的另一个 candidate-only 研究 artifact，登记 `productionReady=false`。它不是本包 candidate-v2 的回滚版本。

## 可信边界

SHA-256 可以证明收到的字节与登记文件一致，但不能证明 pickle/joblib 来源者身份，也不能防御恶意模型代码执行。生产加载前应使用可信传输、权限隔离和显式哈希；不得把接收文件现场计算出的哈希再写回为期望值。

