# Stage 点燃证据：41410864 与 41412533

两次实际初始 Solver screen 共 192 条输出已由原 vendor 解释器在 CPU 上重放。结果是：无 grammar 的 screen 只有 1/96 可解析、0 full-pass；`dsl_grammar_v1` 为 96/96 可解析、1 full-pass，产生一个真实 mixed-reward 组并完成一次 binary-only Solver 更新。唯一成功任务是 identity。需要改变 tape 的 stage 尚未出现成功；这些 screen 没有证明最终 target transfer 或 VERGE 增益。

## 数据与独立核验

原始运行保存在远端 selfgrok 的 `outputs/verge_minimal_20260908/runs/stage_screen_retry1_41410864` 和 `runs/stage_grammar_v1_41412533`。本目录 `stage_ignition/job_<id>` 收录各自 SUMMARY、screen_progress、frozen_screen、12 个 raw sample JSON；grammar 作业另收录 `one_binary_update_proof.json` 与小型 `pilot_state.json`。没有下载完整 checkpoint，没有读取 held-out/test 模型结果。

`stage_ignition/analyze_screens.py` 独立验证 raw token IDs 的长度、原始 suffix 与 format prefix 的拼接、8 条组内 binary reward、full-pass、mixed 判据和汇总统计。两次 screen 的任务顺序、实例 ID、prompt 文本/token IDs、seed、cap、初始 adapter SHA 均逐组相同。9 个 concise stage，加 identity / append_R / append_B 的 official 版本，共 12 组；每组 8 条，普通 cap 768、target_small2/3 cap 2048；temperature=1、top_p=1、top_k=0。全部 screen 来自同一个未更新初始 Solver，更新证明在全部 screen 结束后单独执行。

原 vendor parser SHA256 为 `fa8bb2a00447c0c0f73f5bec34759d60fea08c5c67e99ff0a06850e421acdb72`。CPU replay 对 192 个程序、2848 次测试执行全部通过：parse-valid、binary reward、测试 ID、输入、pass bit、路径步数一致。两个 wrapper 对最终输出不匹配的 reason 分别命名 `output_mismatch` 与 `exact_output_mismatch`；审计显式登记 125 次这一诊断别名，不放宽成功判据。首次因名称不同失败的 `cpu_replay.log` 保留；修正审计比较后的成功日志为 `cpu_replay_verified.log`，完整逐程序诊断在 `cpu_replay.json`。

## 实际计数

| 指标 | 41410864：无 grammar | 41412533：dsl_grammar_v1 |
|---|---:|---:|
| 原始模型输出 | 96 | 96 |
| 可解析程序 | 1 | 96 |
| binary full-pass 程序 | 0 | 1 |
| mixed-reward 组 / 12 | 0 | 1 |
| 实际 optimizer step | 0 | 1 |
| generated tokens | 62,977 | 10,089 |
| 到达长度 cap 的输出 | 33 | 0 |
| 逐测试通过 / 1424 | 0 | 16 |
| 模型 sampling 秒数之和 | 275.14 | 106.36 |

逐测试通过率只用于这里解释 saved 程序行为；实际 Solver reward 仍是全部 case 同时通过的 binary 值。

| Stage / prompt | 无 grammar parse / success / tokens | Grammar parse / success / tokens |
|---|---:|---:|
| identity / concise | 0 / 0 / 3522 | 8 / 1 / 545 |
| append_R / concise | 0 / 0 / 3792 | 8 / 0 / 1074 |
| append_B / concise | 0 / 0 / 4720 | 8 / 0 / 1216 |
| replace_R_to_B / concise | 0 / 0 / 4236 | 8 / 0 / 643 |
| replace_RB_to_YG / concise | 0 / 0 / 3123 | 8 / 0 / 218 |
| prepend_R / concise | 0 / 0 / 3595 | 8 / 0 / 1076 |
| prepend_RB / concise | 0 / 0 / 5595 | 8 / 0 / 500 |
| target_small2 / concise | 0 / 0 / 6468 | 8 / 0 / 1172 |
| target_small3 / concise | 0 / 0 / 12360 | 8 / 0 / 1251 |
| identity / official | 1 / 0 / 6108 | 8 / 0 / 737 |
| append_R / official | 0 / 0 / 5111 | 8 / 0 / 1046 |
| append_B / official | 0 / 0 / 4347 | 8 / 0 / 611 |

每行的 parse 与 success 分母都是 8。这是初始可点燃性探测；采样数相同但生成 token 开销不同，不是按相同训练预算完成的 VERGE 对照实验。两个 prompt 版本各 8 条不足以判断哪一种措辞总体更好。

## 程序语义解释

Grammar 的首个 `START ... NEXT` 目标为：

| 首 route | 数量 / 96 | 直接行为 |
|---|---:|---|
| start | 63 | 立即回到自身，1000 步上限拒绝 |
| NONE | 21 | 立即拒绝 |
| end | 9 | 不修改 tape 即结束 |
| n1 | 2 | 进入中间节点，两个程序都循环 |
| n0 | 1 | append_B 程序进入中间节点，仍错误 |

84/96（87.5%）程序在入口就自环或拒绝。即使后面声明了许多 painter/puller，这些节点也不可达。所有 grammar 失败测试共 1408 次：956 次步数上限、327 次 NONE、125 次输出不符。没有证据支持把这些零奖励解释为该 benchmark 不可解；CPU oracle 已独立证实任务可解，当前障碍主要是模型生成的入口路由与后续 tape 操作。

唯一成功见 `job_41412533/samples/00_identity_concise.json` 的 sample index 7（从 0 计数）。模型实际生成 suffix 为 `end\nEND end\n\`\`\``，与所有任务通用的既有 format prefix 组成：

```manufactoria
START start:
    NEXT end
END end
```

它对 16 个不同 train 输入全通过，并保留每条原始 tape；它不是恒定输出程序。通用 wrapper prefix 不包含完成该 identity 所需的 `end` 选择。该程序仅在事后解释结果，没有被加入训练 prompt 或作为 oracle 示范。

`append_R / concise` 的 8 条中，5 条入口到 start、2 条到 NONE。剩余 sample index 1 到 n1；其所有中间 route 在 n0–n4 内循环，没有任何边到 end，16/16 超时。`append_R / official` 的 7 条入口到 start，剩余 sample index 7 到 end，原 tape 未添加 R，16/16 输出不符。合并两种 prompt 的 16 条 append_R 输出：12 个 start、2 个 NONE、1 个闭环 n1、1 个直接 end；不存在进入有效 append 路径的样本。

仅有的首 route n0 是 `append_B / concise` sample index 0。它先 PULLER_RB 消耗输入，再在一些路径补色；输入 B 的执行结果不是要求的 BB，14 个 case 输出不符、2 个超时。与入口自环不同，它确实执行 tape 操作，但尚未正确维护 FIFO 输入与输出。

无 grammar 的 95 个 parse failure 主要为 invalid node declaration（75），其次 invalid puller route（8）、invalid NEXT route（3）、未定义节点（8）、错误 puller condition（1）。唯一可解析程序见 identity / official sample index 3；程序声明 `END end`，却把一些 route 指向大写 `END`，10 个 case 报 `Unknown node: END`，另 6 个到 NONE。它还用 puller 消耗原 tape，未实现 identity。

由此可以形成一个可测试的实现假设：grammar 要求中间声明使用连续 n0,n1,…，而 v1 通用说明只说 unique node IDs、official 示例还使用其他命名。模型在受限解码后频繁选择 start/NONE，可能与这一表述不一致有关。当前数据不能单独证明原因；因此新增 **names-only prompt view v2** 作为下一次有界 screen 的候选，而不是更改正在执行的 v1 round。v2 逐字增加通用 canonical naming 说明，允许任意合法 start/end/NONE/中间节点路由及循环，不指定 START 接谁，不给任务解。

## 真实 binary 更新证明

Grammar identity 组 reward 为 `[0,0,0,0,0,0,0,1]`。`one_binary_update_proof.json` 记录真实 native-suffix policy 更新：545 个 tokens、545 个 nonzero-advantage tokens、loss `-0.06098948728053941`、optimizer step 1、parameters_changed=true、beta=0、weight_decay=0、binary_reward_only=true。loss 使用与采样相同的 grammar mask。`runtime/hf_backend.py::update` 检查 finite gradient，并比较更新前后实际 LoRA 参数；没有将语法或部分测试通过率作为 reward。

只读远端哈希复核也证实保存了不同权重：原始 adapter `9596df4348ee3d959661bf451864f2b8dafe454212bd16928a6774eba5f5a321`，proof adapter `efb627fd285cfdffda71635128cb7a1e0e9cd04dbc8d0146b9b0da9ac55f2df8`。原始 adapter 哈希与 screen 冻结值相同。`round_base_changed=false`，这次 proof 没有替换 round 的初始 Solver。这证明 binary mixed 组能够驱动真实参数更新；不等于更新改善了 stage 或最终 target 的成功率。

## 实际成本与复现

`stage_ignition/sacct_41410864_41412533.txt` 显示两个作业都 COMPLETED、exit 0:0，各分配 1 B200、4 CPU、64G。

| Job | 开始–结束（sacct 原记录） | 实际 wall 秒 | B200 小时 |
|---|---|---:|---:|
| 41410864 | 2026-09-08 13:25:41–13:31:21 | 340 | 0.09444 |
| 41412533 | 2026-09-08 13:34:41–13:37:55 | 194 | 0.05389 |
| 合计 | 两次 screen 和一次 proof update | 534 | 0.14833 |

batch step MaxRSS 分别 1,950,280K、2,781,288K；这只是 Slurm 报告的 batch step 指标，不代替整个 Python/CUDA 作业的显存峰值。两个作业均在用户 4 B200 / 192G 上限内。此独立审计只使用 CPU，未提交 GPU。

复现入口：在远端 selfgrok release 中运行 `bash evidence/stage_ignition/RUN_REPLAY_CPU.sh`，会读取本地保存的 screen raw，执行 vendor CPU replay，再打印原/proof adapter SHA；输出为 `cpu_replay.json`、`cpu_replay_verified.log`、`adapter_hash_verification.txt`。原始屏幕日志和 checkpoint 不改写。

冻结器第 7 项接口测试也已在 round 部署后复跑：`benchmarks/generated/pilot_view_v1/freeze_cpu_tests_round_deployed.log` 为 7/7 通过。新 names-only view 的生成器为 `benchmarks/freeze_naming_prompt_v2.py`，冻结输入为 `benchmarks/generated/naming_prompt_v2/screen_tasks.json`；它保留 12 项原顺序、每组 8 条、seed、cap、全部 ground_truth、instance tuple，只添加同一命名说明。此报告没有读取或推断正在运行的 round 结果。
