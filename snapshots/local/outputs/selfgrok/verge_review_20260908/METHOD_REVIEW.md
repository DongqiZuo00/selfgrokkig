# VERGE 独立方法复核
2026-09-08；依据：本次原始设计全文、用户本轮澄清、当前工程证据。没有使用旧会话总结或自动记忆作为科研依据。

**结论：该方案有可检验的研究核心，但目前方法定义未闭合，真实工程也不是这份设计的实现。** 固定 target 上、相对同 base direct branch 的课程效果，是合理的评价对象；binary Solver reward 与课程/模型选择的信号分开，也使干预对象较清楚。但“首次成功前非退化”“hint 保证存在可学 stage”“分格允许绕路”均不能直接作为已证结论。旧结果不能用于证明或否定尚未实现的新版闭环。

## 1. 已有原文依据
设计 §1.1：逐测试 exact pass，错误/超时/NONE 为失败；f 为各类通过率的最小值，full pass 为 f=1。parse、四个阈值与 full pass 构成条件；LCP 只记录。
§1.2：注册族、受限变换、hinted target 三类 stage；四项 validator。
§1.3：按 (b,s) 分格，每格一个 incumbent；空格写入、非空格显著改善才替换；最高 rung、再看该 rung rate 选 lead。
§1.4：共享 base，fresh optimizer，beta=0、wd=0，按 generated tokens 分配 B；常数 reward update 精确跳过。
§1.5–1.6：至少两个 branch 的 successes≥q 才切到 full-pass reward；Gamma 对本轮 direct；J+ 要正 gain 和排除零的 paired CI；Challenger 用 archive context 和新增正例累计触发的一遍 rejection-sampling fine-tuning。
§1.7–2：first success、reward switch、learning onset、格子占用、非 lead base 是不同事件；T 或 lead≥0.9 停止。以上是设计规定，不等于工程已满足。

## 2. 独立问题与最小建议
下表建议均未并入原文或正式运行；数值反例来自合成 fixture，不是真实模型效果。

| 原文位置/问题 | 依据与实际影响 | 可执行检验与建议 | 是否需研究决定 |
|---|---|---|---|
| §1.1、§6 “first success 前非退化”过强 | 两个程序在类 A 的 pass fraction 可为 1、0.5，但类 B 均为 0，二者 f 都是 0；即使 f 不同，也可能落在同一阈值区间。必要性不推出区分度、梯度或未来可学性。 | 冻结真实 class 映射后，重算每个 branch 的类向量、阈值占用、零方差率，再检验 P1。建议把非退化改为待测预测。 | 是，修改主张；本轮只给反例 |
| §1.2 hint 的“可学存在性保证”不成立 | 32 个任务中 8 个 hinted 任务每次必成功，其余必失败；无 hint 全失败。两次探针都得到 0.25>0，却没有一个非恒定 reward group。成功率离零不保证有效更新，更不保证无 hint 迁移。 | 分别报告每题组内混合率、实际 optimizer steps、无 hint target 迁移。IID 单题混合概率为 1-p^8-(1-p)^8；使用总体平均 p 代入不适用于异质任务。新增筛选门槛需批准。 | 是；不擅自增加第五项 validator |
| §1.3、§1.6 archive 与 J+ 信用不完全一致 | 空格能保存低分候选，但即时负 Gamma 不进入 J+；其前序课程不会因以后有用而自动得到训练信用。反例：direct=0.4、candidate=0.5、incumbent=0.6，candidate 可进 J+ 但不能替换同格 incumbent。 | 记录正例新增、格子实际变化、回访旧格和 lineage。先测重复局部正例是否推动新状态；不自动增加 hindsight credit 或成功路径回放。 | 修改信用机制需决定 |
| §1.3 lead 与最终 full pass 可冲突 | 合法嵌套 profiles A=(.99,.98,.97,.95,.88,.87)、B=(.99,.98,.97,.95,.89,.01)，同 b=5：lead 选 B，虽然 full pass 仅 .01。 | 合成测试已重现。保留原 lead 规则并同时记录 full pass；如果要改终止输出选择，需单独决策。 | 是 |
| §1.3 静态含环描述能力有限 | 未执行或无用环也能令节点图含环；s 不能证明掌握循环算法。一个 (b,s) 只保留一个 incumbent，仍可能丢掉同格内更有后续价值的分支。 | 用执行轨迹区别图含环与实际循环使用，作为诊断不改变 s。测试零 parse-valid 分母、全条件饱和、完全并列及评估缓存一致性。 | 零分母/终态/并列约定尚需明确 |
| §1.4、§5 compute-matched 范围 | 相同 generated tokens 不等于相同 GPU 时间；不同 L、hint-regret 探针、archive incumbent 重测、Challenger FT 都有额外成本。写“不增加 branch”不能推出不增加算力。 | 分开记录训练生成、阶段探针、最终评估、hint 验证和 proposer 的 token/GPU 时间。各 gate 估算合计 1,425 B200-h：理想连续占用两卡约 29.7 天，未计排队。不是本轮授权。 | 正式预算需决定 |
| §1.5 paired CI 的解释 | 对同一训练产物的 selection-instance bootstrap 只反映评估抽样不确定性，不能替代训练 seed 重复。多轮、自适应 proposal、反复替换导致选择偏差；每格比较还要求匹配实例与 rollout 口径。 | 固定抽样单位、配对 seed 规则、置信水平及多重/序贯比较策略；正式结论使用独立评估。旧代码的 2,000 次两层 percentile bootstrap 是旧实现参数。 | 是，原文未完整冻结 |
| §1.6 跨 seed proposer FT | 跨 seed 共享经验会使其后的 Solver runs 相互依赖；把六条 trajectory 当六个独立重复会低估不确定性。 | 区分 proposer 经验池与独立复现实验单位；P5 在同一冻结 archive 下比较，并将用于评估的新提议与训练正例分开。 | 是 |
| §1.7 “direct 在 first success 前等于 base”需限定事件 | 若指全部训练采样都没有成功，beta=0、wd=0 且不调用 optimizer 时成立；若只指 endpoint 未成功，训练中仍可能已有混合 reward 并更新。first success 也只是采样事件。 | 标记训练/探针/最终采样的事件来源、完整 rollout 序号与费用，不将其命名为 grokking。 | 事件口径需冻结 |

## 3. 两项已确认的未闭合依赖
**测试类映射：缺失。** 用户已确认没有可补充的原始定义。现场 target 生成器、父类、wrapper、verifier、配置和 manifest 均未发现对应的类别列表/映射。生成步骤的名称不构成正式 class；没有按长度、mutation、corner case 分组，也没有把实际测试默认合成一类。

**per-stage Delta / telescoping：缺失。** 设计 §1.5 引用的“修复版”未恢复。相关源码中只有 endpoint 差值、KL 的 ref-logp 等其他 delta；Git 中 VERGE 源码为未跟踪文件，相关版本检索没有给出定义。stage 前后 checkpoint、是否含 target-only tail、固定哪一个 kappa、direct 的进入方式、8/32 样本衔接、最终 telescope 对应的端点差值全部未闭合。本轮没有自造公式、填零或删除新版 context 字段；目前也没有新版 context 实现。

另需明确：无 hint stage 如何适用 hint-regret；test 参数元组的定义；全条件均饱和时 b 的终态；零 parse-valid 时 s；CI/平局约定。当前工程中的取值不能自动成为原文决定。

## 4. P1–P7 能支持多强的结论
| 预测 | 需要收紧的判据 |
|---|---|
| P1 | C-index CI 仅“不含 0.5”也可能落在 0.5 下方，应事先明确正向预测；全删失时不可算有效排序。按轮分层仍不足以提供独立训练重复。 |
| P2 | 看到一条经过 s=1 回退格的成功血统是机制存在证据，不证明分格带来因果收益；有限运行中没有观察到也不证明分格无用。 |
| P3 | “比例更高且无 hint rung 上升”和“从不进 J+”并非互补判据；存在一些 J+ 但不优于注册族时，表中没有清楚的裁定。 |
| P4 | 要匹配提议次数、有效率和成本，不能仅比较已经过不同过滤的正例比例。 |
| P5 | 同一冻结 archive、未参与 FT 的新提议及独立重复；无显著差异不等于等效。 |
| P6 | 有限预算 direct 未成功不是不可能性证明；按结果换 target 会改变预先声明的研究总体，须披露筛选和重新冻结。 |
| P7 | replay 应匹配初始状态、任务多重集、target tail、生成预算与训练 seeds；不显著不能直接证明顺序无价值。 |

## 5. 现场证据与方法判断分开
旧 book_v2 的四臂×六轮共有 96 个端点；保存向量重算全部一致，端点 full pass 全为 0，训练累计 target successes 也为 0。旧 VERGE 臂仍有 5/6 轮 Challenger 更新，这是旧在线 policy-gradient 路径的事实，不是新版 J+ FT 或 grokking 的证据。抽样重放 64 条已保存 completion，binary verifier 结果 64/64 一致；未重新生成模型样本，未读取 sealed held-out/test 结果。

合成规格测试 10 项通过；工程测试 68 项通过。测试类 fixture 仅验证给定映射的算术；archive/Gamma/J+ fixture 仅验证明确文字规则，**不表示正式组件已经实现**。详细结果见 IMPLEMENTATION_AUDIT.md。

## 6. 相邻文献核验的范围
SOAR 的原论文确实把 teacher return 锚定到困难目标上的 student improvement；这支持区别讨论的起点，不自动证明 VERGE 新颖性或有效性。[SOAR，arXiv:2601.18778v3](https://arxiv.org/abs/2601.18778v3) 的作者记录标注 ICML 2026。
RL Grokking Recipe 原论文讨论 dense warm-up、curriculum 与迁移，不能把本文 first success 当作其 grokking 判据。[原论文](https://arxiv.org/abs/2509.21016v2)、[作者代码](https://github.com/sunblaze-ucb/rl-grok-recipe/blob/main/README.md)。本轮未全面复核 §3 其余工作的优先权或全部会议归属；没有据此作唯一性主张。

