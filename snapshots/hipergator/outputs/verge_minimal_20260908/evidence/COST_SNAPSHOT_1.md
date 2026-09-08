# VERGE 实际成本与预算

状态：**snapshot_or_needs_reconciliation**；统计时间：2026-09-08T19:15:56.431413+00:00。

实际保存的 native generation 共 **653,765 tokens**；Slurm 分配 **1.756944 B200 小时**。

| 类别 | 实际 native tokens |
|---|---:|
| solver_training | 464,133 |
| solver_evaluation | 94,005 |
| challenger | 5,530 |
| stage_probe | 90,097 |

| 作业/阶段 | Solver train | Solver eval 新生成 | Q | Stage probe |
|---|---:|---:|---:|---:|
| failed_initial_stage_launch | 0 | 0 | 0 | 0 |
| unconstrained_stage_screen | 0 | 0 | 0 | 62,977 |
| grammar_stage_screen | 0 | 0 | 0 | 10,089 |
| naming_v2_stage_screen | 0 | 0 | 0 | 17,031 |
| round1_preparation | 0 | 33,854 | 2,804 | 0 |
| round1_g1 | 65,536 | 0 | 0 | 0 |
| round1_g2 | 65,536 | 0 | 0 | 0 |
| round1_g3 | 65,536 | 0 | 0 | 0 |
| round1_direct | 65,536 | 0 | 0 | 0 |
| round2_preparation | 0 | 60,151 | 2,726 | 0 |
| round2_g1 | 65,536 | 0 | 0 | 0 |
| round2_g2 | 65,536 | 0 | 0 | 0 |
| round2_g3 | 34,650 | 0 | 0 | 0 |
| round2_direct | 36,267 | 0 | 0 | 0 |

| Slurm allocation | State | Elapsed seconds | B200 小时 |
|---|---|---:|---:|
| 41410512 | FAILED | 0 | 0.000000 |
| 41410864 | COMPLETED | 340 | 0.094444 |
| 41412533 | COMPLETED | 194 | 0.053889 |
| 41413994 | COMPLETED | 532 | 0.147778 |
| 41414619_0 | COMPLETED | 716 | 0.198889 |
| 41414619_1 | COMPLETED | 727 | 0.201944 |
| 41414619_2 | COMPLETED | 738 | 0.205000 |
| 41414619_3 | COMPLETED | 810 | 0.225000 |
| 41415498 | COMPLETED | 212 | 0.058889 |
| 41417194 | COMPLETED | 530 | 0.147222 |
| 41418169_0 | COMPLETED | 453 | 0.125833 |
| 41418169_1 | COMPLETED | 475 | 0.131944 |
| 41418169_2 | RUNNING | 299 | 0.083056 |
| 41418169_3 | RUNNING | 299 | 0.083056 |

这些已命名作业的最大同时分配为 2 GPU、128 GiB；与用户上限 4 GPU / 192 GiB 的比较为 True。这不包含其他项目或未列入的作业，也不是实际 GPU 利用率或进程峰值内存。

训练预算以真实 raw token IDs 对照 phase ledger；通过 EOS 的最后一个 token计入，固定格式前缀、prompt 和 EOS 后 batch padding 不计入。评估仅计算实际原始生成文件，缓存 alias/复制的 fragment 不增加 generation。相同 token 内容来自两次独立生成仍算两次。

Stage proof 复用已计入 screen 的输入 tokens；其 optimizer 计算已经包含在该 Slurm 作业的分配时间内，不再加一份 generation：
- 41412533: 复用 545 tokens，新增 generation 0，来源核验 True。
- 41415498: 复用 822 tokens，新增 generation 0，来源核验 True。

失败作业照录，不删去其耗时。41410512 的 sacct ElapsedRaw 为 0 秒时，只能报告该秒级记录为 0，不能据此推断未被计量的亚秒开销；没有保存的 raw 也不被伪称为已测得模型生成 0。

第一、二轮使用不同 prompt view，其成本分别列出；该报告不把两轮结果当成同一个固定 policy 实验。此前 release 的 GPU smokes、CPU 语法检查以及其他项目不在本次八个主 job ID 的统计范围内。

## 完整性检查

唯一 generation 来源文件：603；核对 evaluation 原始来源：64；缺失预期 scheduler tasks：[]。
全部已执行的来源、预算、计数和 proof 复用核查通过。最终状态另要求每个预期 Slurm task 都已终止。
