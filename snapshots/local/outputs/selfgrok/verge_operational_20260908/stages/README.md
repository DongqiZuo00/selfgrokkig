# 本轮新增：stage validator 与 Challenger RFT buffer

本目录是 2026-09-08 在用户允许补齐操作性定义后新增的 CPU 组件。依据是最新设计 §1.2、§1.5、§1.6；不是此前代码已实现的事实，也不是任何训练或科研有效性结果。全部测试输入、验证库条目、condition rates 和 checkpoint 名称均明确标为合成 fixture。

`operational_stages.py` 保留 registered、tape_transform、hinted_target 三种来源；restricted_transform 是显式兼容别名，结果 JSON 中 canonical_kind 统一为 tape_transform。注册族函数和测试协议由调用方可信注册；proposer 只选择规格。自写变换使用数据 DSL：input、literal、reverse、rotate、slice、replace、concat、contains、equals；没有 eval、exec、属性访问、任意 Python、外部 IO 或循环，深度≤8、节点≤64、输入长度≤1024、输出长度≤4096。显式分布必须声明 alphabet、min_length、max_length、corner_cases。`verge-stage-cases-v1` 按 corner、长度端点和固定 seed 的 32 个样本生成测试；该协议及限制属于本轮新增操作约定。

`validate_stage` 恰有四项检查：可执行、corner 上非常量、完整实例参数元组不与受保护 manifest 重合、hint-regret。无 hint 的第四项记为 **NA**，不会伪造有/无 hint 探针。受保护 manifest 必须由调用方显式提供，缺失即不接纳；其哈希入结果。本轮实例身份定为 **(target-definition SHA256, suite-input-set SHA256)**，排除 split、row ID 等标签；`instance_tuple_from_manifest` 按真实定义和去重排序输入集合构建，换 split 标签不能让相同实例逃过比较。必须由可信适配层从冻结 manifest 构建，不得采信 Challenger 自报的哈希。调用方仍须核查不同 suite 的逐 tape 不相交，完整 tuple 比较本身不替代部分交集检查。不能从模型结果推分组，不能只传抽样子集声称完成全分布隔离。

hint 必须引用 `VerifiedProgramLibrary` 已登记条目：train 来源、非空逐测试全通过记录、测试 ID、程序 SHA-256、verifier 协议 ID 和验证日志 ID。更小实例要求同族且 size 严格更小；相关跨族实例必须在可信注册表的 `related_hint_families` 中提前声明。验证器只收到 family/tier/mutation/distribution，收不到 hint 内容或 hint ID。`render_target_prompt(..., final_evaluation=True)` 拒绝传入 hint 和训练 hint 标记；最终 target 探针与评估使用原始 base prompt。实际程序执行和证据日志解析由工程 verifier 适配层承担；本模块检查已传入记录，不把合成验证记录变成真实程序验证。

原文“32 题 × 8 条，两次”与 §5 每 stage 512 rollouts 有歧义。本轮明确采用 **有 hint、无 hint 两臂各 32×8=256，合计 512**，固定同一组 32 个实例 ID、同一 rollout seed 索引和 binary-full-pass condition。仅判断 `p_with > p_without` 与 `p_without < 1-delta`（默认 delta=.1）。这是新操作性约定，不称从原文无歧义恢复。“两次独立重复各含两臂=1024”不属于默认协议；若以后增加复测，单列预算。混合 reward 组数量只进入诊断，不新增第五道 validator 门，也不声称过门已保证 Solver 能学或能迁移。

`RejectionSamplingBuffer.add_round` 追加全轮经验，只有给定 J+ 且 Gamma>0、paired CI 下界>0 的候选进入正例 buffer；保存 context、base、curriculum、seed、condition、证据 ID 和内容哈希。跨 seed 新增数达到 `n_min` 时 `plan_update` 生成一次 **全部累计正例、单 epoch RFT 计划**。没有在线 Challenger GRPO。计划不执行训练；只有 `complete_update(success=True, checkpoint_id=..., training_log_id=...)` 才消耗相应新增计数。失败保留全部新增数并记录原因；训练期间新到正例不被旧计划成功误消耗。J+ 为空的轮留在 archive 并计数。

CPU 命令（本目录作为工作目录，Python 标准库即可）：

```text
python -B -m unittest discover -s . -p test_operational_stages.py -v
python -B demo_stages.py
```

`cpu_tests.log` 记录实跑的 18 个实质测试：三种来源、分布/DSL边界、泄漏、hint来源与配对预算、最终prompt、J+筛选、跨seed累计阈值、训练失败/并发新增计数及provenance。`demo_stages.json` 只报告合成CPU演示和未执行的RFT计划。GPU使用为0，未提交作业，未修改旧工程。
