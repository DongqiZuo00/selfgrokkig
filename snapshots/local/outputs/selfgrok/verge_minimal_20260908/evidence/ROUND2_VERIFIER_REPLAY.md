# 第二轮实际训练 prompt / verifier 抽查

对数组 41418169 的 g1、g2 实际训练记录抽查 **5 组、40 条模型输出、528 次 case 执行**，原 vendor CPU 重放全部一致。被抽查的阶段确实使用 `naming_round_v2/catalogue.json` 中的 **concise 命名版 rows**，不是 official prompt。未发现这些组的 prompt 接线、测试版本或 binary verifier 不一致。

这份报告只覆盖固定训练快照和五组重放，不代表完整第二轮结果。未读取 held-out/test 模型结果，也未读取阶段或最终 target evaluation 文件；原训练、checkpoint 和运行契约均未改动。

## 快照与选择

原始执行目录为 `runs/round2_g1_41418169_0/execution` 与 `runs/round2_g2_41418169_1/execution`。LAUNCH 中实际子作业号分别为 41418185、41418186。

取到的只读 training_journal 快照：g1 为 167 条记录、末序号 166；g2 为 164 条、末序号 163。各含 39 个完成的 generation group。完整 hash 链均验证通过；两份快照内正奖励组数和 actual_model_update 事件数都为 0。这是快照事实，运行中的作业随后可以继续产生新记录。

抽查规则已固定在 `round2_verifier/selection.json`：优先含正奖励的完整八条 stage 组；若没有，则按 journal 顺序，为每个不同 stage ID 选首个完整八条组，最多五组。当前快照没有正样本可优先选，最终覆盖两个 branch 的五种不同 stage。只对这五组做 vendor 重放，没有重跑数千条零奖励样本。

| Branch / phase | Stage | 原始 raw 文件 | n / cap | tokens | parse / binary success | case pass / total |
|---|---|---|---:|---:|---:|---:|
| g1 / 1 | prepend_RB | 000000_g1_segment_1.json | 8 / 2048 | 995 | 8 / 0 | 0 / 128 |
| g1 / 2 | target_small3 | 000017_g1_segment_2.json | 8 / 1141 | 1302 | 8 / 0 | 0 / 104 |
| g1 / 3 | prepend_R | 000024_g1_segment_3.json | 8 / 2048 | 1788 | 8 / 0 | 0 / 128 |
| g2 / 1 | replace_RB_to_YG | 000001_g2_segment_1.json | 8 / 1910 | 1183 | 8 / 0 | 3 / 128 |
| g2 / 3 | target_small2 | 000027_g2_segment_3.json | 8 / 2048 | 1706 | 8 / 0 | 0 / 40 |
| 合计 | 五组 |  | 40 条 | 6974 | 40 / 0 | 3 / 528 |

1141 / 1910 是实际 journal 冻结的该组生成 cap，与原始 raw 完全一致；执行器会按该段剩余 generated-token 预算限制 cap。没有把这些组重新按 probe 的 768/2048 cap 解读。

## 实际取用链路

两份 LAUNCH 明确绑定 `benchmarks/generated/naming_round_v2/catalogue.json`，SHA256 为 `4fbe279cfed264dc14f9a4fca8072d8c9adaf841bff33fda38587d5d6d2649d7`；target rows 为该目录的 `target_rows.json`，SHA 为 `1e1407c3f529cd60da01ebf9888dc1635dae9da405c40a0d050139d77ade9420`。计划文件 SHA 为 `c6bae12f21c724db985666fb6b41139cfbd4309a6008fed1ef27d66197c6c46f`。审计重新计算并匹配这些文件，以及 runtime contract 中全部 16 项源文件哈希。

`runtime/execute_round.py` 在选到 stage 时从 `stage_by_id[phase["stage_id"]]["rows"]` 取训练行。独立核查计划 phase 对应的 stage ID、该 stage 的 kind/spec 和实际 raw 的 instance ID；新 catalogue 中 `rows` 与 `rows_by_variant["concise"]` 相等。五组 instance ID 均为 `__compact` 后缀。

对 prompt 的验证不限于文件名：每组 raw 保存的 **完整 chat prompt** 与此前已验证的 names-only v2 concise probe 中相同 instance ID 的完整 prompt 逐字相等；还验证其包含新 catalogue 的 user 内容、通用命名段落恰好一次、hint=None，而 official 版本的 messages 与它不同。因此这五组实际模型输入没有落回旧版或 official 说明。

一个序列化细节已经单独处理：launcher 将源 `configs/round2_runtime_contract.json` 读为 JSON 后写入运行目录，CRLF 变为 LF，所以复制文件字节哈希不同。审计先核实 LAUNCH 声明的原始文件字节 SHA，再核实运行目录副本的 JSON 内容完全一致；没有将这种换行差异误判为运行契约改变。

## Binary 与逐测试复验

`round2_verifier/replay_training.py` 逐组核实：原 raw 文件 SHA 等于 journal 的 generation_source；raw seed、n、cap、optimizer_steps_before 与 journal 一致；journal rollout 的 token IDs、完成文本、reward 和 verifier_digest 与 raw 及冻结 ground_truth 一致。verifier_digest 重新由 `{ground_truth, task_kind:"stage"}` 计算。五组都来自 stage pool，不是 target 混入或 boundary drain。

CPU 使用原 vendor parser，SHA256 `fa8bb2a00447c0c0f73f5bec34759d60fea08c5c67e99ff0a06850e421acdb72`。逐条重放核对 parse_valid、binary reward、case 的输入/test ID/pass/执行步数。每个 reward 都重新检查为“所有 case 同时通过”。共 528 次执行全部与原记录一致。

失败分解为 363 个最终输出不符、146 个步数上限、16 个路由 NONE，另 3 个 case 通过。catalogue helper 的 `output_mismatch` 与实际 backend 的 `exact_output_mismatch` 是同一诊断的名称差异，共 363 次，审计明确记录这项别名，其他字段严格比较。replace_RB_to_YG 的 3 个单 case 通过仍保持 binary reward=0；没有部分测试奖励混入。

这些证据表明，被抽查的零奖励来自程序没有完整完成各自任务，而非这些组误用了 official prompt、错误测试行或错误 binary 聚合。它们不能说明没有抽查的样本、后续阶段或整个 round 的最终表现。

## 复现与产物

运行 `bash evidence/round2_verifier/RUN_REPLAY_CPU.sh`。该脚本只读已固定的 training journal/raw 快照与 source contract，先验证选择，再做原 vendor CPU 重放。结果为 `round2_verifier/cpu_replay.json`、`cpu_replay.log`、`selection_cpu.log`；原五组 raw、LAUNCH、runtime contract 副本和 journal 快照分列在 `g1/`、`g2/`。全部产物已同步本地和远端 selfgrok。没有提交新 GPU 任务或更改正在运行的文件。
