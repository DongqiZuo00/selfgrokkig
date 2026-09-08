# VERGE 本轮最小实验的完成判据

本判断重新通读了原始 `outputs/selfgrok/verge_review_20260908/evidence/VERGE_own_method_design.md` 全文，并核对当前 `runtime/execute_round.py`、`round/round_cli.py`、`runtime/prepare_ignition_round.py`、`runtime/round3_results.py`、`runtime/audit_real_scores.py`。依据是原文 §1.4–1.7、§2 的运行与记录定义，§4 的待检验预测，以及当前明确标注的开发协议；不是此前会话的科研结论。

**结论：完整跑完且审计通过的一轮，可以验收为“完成了含真实 Solver 更新的有界 VERGE 流程实验”。不要求 Gamma 为正或出现 target success 才能验收流程；二者若仍为零，就是本次实验的实测负结果。一次 stage update 本身还不是整轮完成。**

审阅时的实际磁盘快照保存于 `THIRD_ROUND_PROGRESS_AT_REVIEW.json`：array 41420288 的 g1 已记录一次实际更新和一次 stage success，尚在训练；g2 仍在训练，g3/direct 尚未开始。当前不能提前宣布整轮完成。

## 完成所需的最小证据

1. 保留真实 Q JSON 与冻结计划，声明 g1 第一阶段由 Q 在两个已有 mixed 观察的阶段中选择这一**人为开发先验**；其余八位置为原菜单。四项 validator、无 hint 的 target 测试、同一初始 base 和 fresh optimizer 均与计划一致。
2. G3 与 direct 全部完成：每分支 65536 generated tokens，共 262144；课程三阶段与 target-only tail 均有真实 native-token 账本。至少一个 curriculum mixed binary group 确实调用 optimizer、改变参数并留下真实 checkpoint；同奖励组精确跳过。不能用仅保存文件、计数器增加或 probe 中的独立更新代替本轮 branch 的实际更新。
3. 完成各阶段 P8 与最终 P32 的无 hint target 评价；仍用冻结的 8 个 selection 实例、每实例全部 36 测试与六类，以及原始 min-per-class 分母。错误/超时计失败，样本不丢弃。缓存只引用相同权重与采样合同下的原始记录；本轮复用的旧 baseline 256 条计为引用证据，新增生成量为零。
4. 完成 merge、Gamma/paired CI、包含 tail 与 P32/P8 端点衔接的 Delta/telescoping、archive/lead、J+ 与 buffer/RFT 决策、事件及 lineage 输出，并通过原 raw 文件/权重/manifest/SHA/预算核对和独立评分审计。逐阶段 Delta 与类别映射按已声明的操作性补充执行，不称为从原文缺失引用中自动恢复。

## 若三条 Gamma 仍为 0，最终 target rate 仍为 0

| 可以陈述的实测事实 | 不能据此陈述的研究结论 |
|---|---|
| 真实 Q 提案接入执行器，binary stage reward 能产生真实 Solver 参数更新，checkpoint/日志链路可运行。 | 已解决 target 的可点燃性，或 stage 更新已经迁移到 target。一次参数移动也不证明 stage 泛化改善。 |
| 完成 matched-token direct 对照与固定 target 评价；本轮完整课程相对 direct 的所选 condition gain 为零。 | VERGE 有效，frontier gain 在 first success 前提供了有用的非退化搜索信号，或它能预测后续成功。 |
| J+ 为空，buffer 没有新增正例，因此按原文规则不做 Challenger RFT。这个“不更新”分支被真实执行。 | Challenger 已学到课程选择能力，真实跨 seed RFT 学习链路已验证，或 P5 已成立。 |
| archive 按实际 rung/结构与替换规则处理每个最终分支，输出实际 lead。 | 分格能支持有效绕路、loop 必要、发现的顺序优于 pooled mixture、hint 或自写变换有优势。 |

两个易混淆处必须按实际记录分别报告：**stage success 不等于 target first success；最终 target rate 为零，也不自动排除训练或中间探针曾出现 target success。** first-success 要查所有相关无 hint 原始记录与训练事件，不能用缺少全局顺序的 `null` 字段充当“从未成功”的证据。同样，Gamma=0 不意味着每段 Delta 都为零，暂时收益可能在尾段消失；Gamma=0 也不意味着 archive 只有一格，结构描述子可能产生新格。

若所有最终分支 target success 均为零，本轮不会触发 full-pass reward switch，lead 也未达到 §1.7 的 0.05 learning-onset 门槛。P1–P7 仍须按各自原定比较与证伪条件检验；本轮的 8-instance、单 seed、有首阶段人为先验的结果不完成 Gate 0–3，不验证开放任务生成优势，也不验证持续多轮、多 incumbent 的 archive 搜索与跨 seed proposer 学习。当前 scorer 每轮从 base 建立 archive/buffer，历史进入 context；这一工程事实也不能表述成全部长期闭环已经跑通。

建议最终报告用语：**“完成一轮含真实二元奖励 Solver 更新的 VERGE 开发流程实验；所有分支、目标评价和评分审计完成。本轮三条课程的 Gamma 均为零，最终 target rate 为零，未观察到完整课程的 target 迁移收益；J+ 为空，按规则未更新 Challenger。首阶段使用了公开的人为点燃先验，方法有效性与原定预测仍未验证。”** 其中“所有分支完成”和数值只在最终真实记录确认后填写；first-success、Delta、archive 格数另按实测报告。
