# 第二次真实 Challenger 审查与下一次最小干预

审查对象是 `runs/round_prepare_named_41417194` 的实际 `challenger_sample_0.json`、`challenger_context.json`、`challenger_schema.json`、`verified_screen.json`、`READY.json`，以及命名探针 `41415498` 的原始样本和更新证明。此目录保存 JSON 内容副本；PowerShell 文本传输可能改变换行，远端原件的字节 SHA 单独保存在 `remote_source_sha256.txt`。当前 array `41418169` 的提案和作业保持不变。

已确认的事实：

- Q 使用原始独立 Challenger adapter，实际 seed 为 20269908，生成 2726 tokens，用时约 63.24 秒。上下文长 16477 tokens。
- 指令已经明确写出：Solver 只在成功/失败混合的 binary reward 组上更新，全 0 和全 1 组精确跳过，并要求选择/排序时使用 mixed 证据。不能把本次选择归因于缺少这句规则。
- 新命名 view 的原始 Solver baseline 是实际 256 条无 hint 评价，rho = `[1,0,0,0,0,0]`，full pass = 0。
- concise `identity` 和 `append_R` 分别出现 1/8 binary success。两组都是真实 mixed；`identity` 的 822 个 native tokens 产生了真实 optimizer step 和参数变化。证明 checkpoint 独立保存，没有替换各分支的原始共同 base。
- 其余七个 concise stage 各为 0/8，三个 official 变体也各为 0/8。这表示这次有限探针没有观察到成功，不能据此宣称它们不可能成功。

Q 实际选择为：

| 分支 | 第一阶段 | 第二阶段 | 第三阶段 |
|---|---|---|---|
| g1 | prepend_RB | target_small3 | prepend_R |
| g2 | replace_RB_to_YG | target_small3 | target_small2 |
| g3 | target_small3 | prepend_R | prepend_RB |

九个位置均未选择已有 mixed 观察的两项。它选择了语义上贴近目标的阶段；仅凭受 schema 约束的 JSON 输出，无法确定选择动机。上下文较长且历史信息较多是可疑因素，但没有对照证据足以把它认定为原因。再次加一句相同的软提示不能确保改变这个具体选择问题。

本次建议并已由主 agent 采纳为有条件的下一轮开发策略：**只限制 g1 的第一阶段，由真实 Q 在 `identity` 与 `append_R` 中自行选择；其余八个位置保留原九项菜单。** 新 schema 使用 `g1.prefixItems` 表达首位置限制，完整 stage 对象仍由真实 Q 生成，任何历史 Q 输出都不被重写。该先验来自与训练相同 concise prompt 的实际 mixed 观察，不使用 held-out/test 模型结果。

这是一项公开的人为提案先验，改变下一轮开发提案的采样空间。它不改变 binary reward、四项 validator、目标问题、36 测试与六类、三种 stage 来源接口、G3/L3、共同初始 base、direct 对照、target-only tail、预算或 Gamma/Delta 定义。它不能证明无约束 Challenger 已能自主发现有效课程，也不能保证新的随机训练组再次 mixed 或 Gamma 为正。正式研究范围保留，尚未验证的主张继续保持待验证。

生产入口是新文件 `runtime/prepare_ignition_round.py`。它要求 `round2_result` 完整且实际 curriculum update 数仍为零，才启动这项条件干预；历史未完成时不会加载 Q。原命名 view 的 256 条 base 记录按照相同权重与完整 baseline 合同复用，逐个检查原 raw 路径与 SHA；新准备阶段增加的 baseline rollout/token 数都是零。新 READY 在独立目录引用复制的 fragment，fragment 内记录仍指向旧作业的真实 raw。冻结 scorer 的“引用证据总量”与本次新增生成量需要分别报告。

最小调整没有改变初始化权重，也没有把 probe 的成功程序变成 target hint、监督训练答案或预先代填的课程。若正在执行的第二轮自己出现真实 curriculum update，则这项仅用于点燃的后续干预没有必要，入口会拒绝继续。
