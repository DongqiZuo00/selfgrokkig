# VERGE 三轮累计成本与预算

状态：**complete**；UTC 统计时间：2026-09-08T19:51:04.700195+00:00。

累计保存的实际 native generation：**1,039,681 tokens**。Slurm 分配：**2.542500 B200 小时**。原两轮 `COST_AND_BUDGET.md/json` 保留。

| 类别 | 实际 native tokens |
|---|---:|
| solver_training | 786,432 |
| solver_evaluation | 155,060 |
| challenger | 8,092 |
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
| round2_g3 | 65,536 | 0 | 0 | 0 |
| round2_direct | 65,536 | 0 | 0 | 0 |
| round3_preparation | 0 | 0 | 2,562 | 0 |
| round3_g1 | 65,536 | 61,055 | 0 | 0 |
| round3_g2 | 65,536 | 0 | 0 | 0 |
| round3_g3 | 65,536 | 0 | 0 | 0 |
| round3_direct | 65,536 | 0 | 0 | 0 |

## 第三轮 baseline 复用

新增 baseline 为 **0 tokens / 0 rollouts**。256 个 slots、32 个原始 generation chunks 的 token IDs、completion 内容及 SHA 全部与 41417194 保持一致。原先 **60,151 tokens** 只在第二轮 preparation 计一次。第三轮复制的 fragment、original_execution 和逻辑评估别名均未另计 generation。

## Slurm 分配

| Allocation | State | Seconds | B200 小时 |
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
| 41418169_2 | COMPLETED | 488 | 0.135556 |
| 41418169_3 | COMPLETED | 475 | 0.131944 |
| 41419999 | COMPLETED | 147 | 0.040833 |
| 41420288_0 | COMPLETED | 865 | 0.240278 |
| 41420288_1 | COMPLETED | 489 | 0.135833 |
| 41420288_2 | COMPLETED | 490 | 0.136111 |
| 41420288_3 | COMPLETED | 472 | 0.131111 |

已列入作业的峰值同时分配为 **2 GPU / 128 GiB**；用户上限为 4 GPU / 192 GiB。这是 allocation，不能解读为实际 GPU 利用率、进程峰值内存或所有其他作业的总资源。

失败分配照录，包括 41410512。其 ElapsedRaw=0 是 sacct 的秒级记录，未虚构亚秒耗时；没有保存的 raw 不被误称为测得模型 generation=0。

## 计数与验证范围

逐个实际 generation 来源核算 native token IDs，包含 EOS，排除 EOS 后 batch padding、prompt 和固定格式前缀。独立采样即使输出相同仍分别计数；缓存 alias 或复制 fragment 只引用原来源。Stage proof 复用 545/822 个 screen tokens，不产生额外 generation；更新计算已包含在其作业的分配时长内。

第三轮沿用第二轮 Solver prompt view / grammar / 原始基线，并增加已声明的 Challenger 第一阶段 proposal prior。这里仅比较成本，不把不同 prompt view 或 proposal policy 的三轮合并成同一效果实验。此前 release 的 smokes、CPU-only 语法检查、其他项目和未明确列入的作业不在 GPU 作业统计中。

核验 876 个独立 generation 来源；missing scheduler tasks=[]；全部指定任务终止=True。

本报告的 CPU 日志 `evidence/COST_V3_AUDIT_TOOL_CPU_TESTS.log`：18 项，完成标记=True。这些全部是**成本/来源审计工具测试**，包括此前 v2 的 12 项；不并入 Solver、Challenger 或训练路径测试数量。训练路径测试由相应执行报告单独统计。

审计问题：

无已发现的计数或来源问题；最终状态还要求第三轮 array 身份已知、全部任务终止且 baseline 复用核验通过。

复现使用新文件名，保留已有报告：

```bash
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B -m unittest discover -s runtime -p 'test_account_round_cost_v*.py' -v
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/account_round_cost_v3.py --round3-array 41420288 --output evidence/COST_THREE_ROUNDS_RECHECK_NEW.json --markdown evidence/COST_THREE_ROUNDS_RECHECK_NEW.md
```
