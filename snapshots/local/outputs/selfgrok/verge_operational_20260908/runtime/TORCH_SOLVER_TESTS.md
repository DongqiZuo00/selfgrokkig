# 真实 Torch 更新的 CPU 回归

`solver_update.py` 的 `phase_boundary_masks`、`disable_training_dropout`、`chosen_masked_logp`、`token_loss`、`train_step` 五个函数，来自本轮此前独立审计副本 `verge_review_20260908/review_copy/src/verge_repair_training.py`。这些函数原样保留作为回归对象；这不表示旧版 loss-token 预算协议被采用。

新入口 `binary_solver_train_step` 仅接收完整 8 条 rollout、binary reward、beta=0、weight_decay=0，使用全部生成 action 的 loss mask，不截断响应；generated-token 准入和预算由 `generated_budget.py` 与上层驱动负责。`make_fresh_solver_branch` 只克隆模型权重，新建空 AdamW 状态和新 scheduler。没有 KL teacher、KD、路由或 MPC 训练路径被新入口启用。

`test_torch_solver.py` 使用 CPU 上随机初始化的 `TinyLM`，直接调用真实 Torch 和复制的真实 `train_step`。六项测试检查：binary advantages；已产生 AdamW 动量后全 0/全 1 reward 组的参数、optimizer、scheduler 精确不变；mixed group 确实移动参数；各分支 fresh optimizer；beta/wd 限制；first success 前 direct 分支重复更新仍等于 base。TinyLM 只用于工程回归，不是研究 backbone，不构成训练有效性证据。

本地无 Torch 时只完成 AST 语法检查，见 `torch_solver_local_syntax.log`。应在远端现有 Torch 环境、CPU 上实际执行：

```text
python -B -m unittest discover -s runtime -p test_torch_solver.py -v
```

脚本无模型下载、CUDA 调用、作业提交或原 checkpoint 修改。若 Torch 不可导入，unittest 明确 skip，不能把 skip 当作上述六项执行通过。
