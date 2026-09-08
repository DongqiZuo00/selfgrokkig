# Names-only stage screen：41415498 独立复核

新命名说明后的 96 条真实输出已通过原 vendor CPU 重放：96 可解析、2 个 binary full-pass、2 个 mixed 组。首 route 到 start/NONE 的程序由原 screen 的 84/96 变为 2/96，94 条改为进入 n0。新增的一个成功来自 append_R；这证明当前初始 Solver 至少采样到了一条正确的非 identity tape 变换程序。**2/96 对 1/96 不构成显著提升或 target gain 结论**，这也不是一次完整 VERGE curriculum/direct 对照。

## 采样契约核对

比较作业为同一 selfgrok release 的 `stage_grammar_v1_41412533` 与 `naming_v2_41415498`。原始新 screen 的 SUMMARY、progress、frozen_screen、12 个 samples 文件、单步更新证明与小型 pilot_state 已保存到 `evidence/naming_v2/job_41415498`。未下载权重文件、未读取 held-out/test 模型结果。

`evidence/naming_v2/replay_naming_v2.py` 对每组独立断言：

- 原初始 Solver 路径及 adapter SHA 一致，都是 `9596df4348ee3d959661bf451864f2b8dafe454212bd16928a6774eba5f5a321`；每个 group 的 optimizer_steps_before 都是 0。
- 相同的 12 项 stage/variant 顺序、8 samples/group、seed、cap（普通 768，small target 2048）、temperature=1、top_p=1、top_k=0、format prefix、DSL decoding policy。
- 冻结 row 的 ID、ground_truth 及所有非 messages 字段逐项相等，输入均来自 train。新实际 chat prompt 恰好把旧 user 内容替换成“旧内容 + 两个换行 + 已冻结的同一通用命名段落”；其余 chat wrapper 和 format prefix 完全一致。
- grammar SHA `46ee784f8dad729fccfe290dd80560a5f14f7b822acffee2df1467f752ef9e5e`、tokenizer JSON SHA、完整词表 SHA、vocab_size=131072、EOS=[2]、max_nodes=32、XGrammar/Transformers 版本一致。

两次 metadata 并非字节相同：worker 启动耗时不同；新 metadata 增加 `mode=solver_dsl`、`consumed_prefix=原 format prefix`、`structured_const_expansions=0`、`json_schema_path=null` 四项说明。独立读取当前 `decoding/grammar_service.py` 并与该模块维护者核对后确认，它们来自增加 JSON schema 分支时的诊断字段：DSL 仍编译原 `build_grammar`，消费原 PROGRAM_PREFIX，不执行 JSON const 展开。审计区分缺失键与 null，记录这些明确差异，没有以“整份 metadata 一致”掩盖它们。

CPU 使用原 parser SHA `fa8bb2a00447c0c0f73f5bec34759d60fea08c5c67e99ff0a06850e421acdb72`，重新执行旧、新各 96 个程序和各 1424 个 case。全部 parse、binary、逐 case 输入/test ID/pass/steps 与保存记录一致。延用已说明的唯一诊断名别名：catalogue 的 `output_mismatch` 对应 backend 的 `exact_output_mismatch`，旧 125 次、新 867 次；成功判据保持严格 exact output/full-pass。

## 观测结果

| 指标 | 41412533，v1 | 41415498，names-only v2 |
|---|---:|---:|
| 原始输出 / 可解析 | 96 / 96 | 96 / 96 |
| binary full-pass | 1 | 2 |
| mixed 组 / 12 | 1 | 2 |
| 实际 optimizer steps | 1 | 1 |
| generated native tokens | 10,089 | 17,031 |
| 到 cap 的输出 | 0 | 0 |
| 逐 case exact pass / 1424 | 16 | 44 |
| identity concise 成功 / 8 | 1 | 1 |
| append_R concise 成功 / 8 | 0 | 1 |
| 其余十组成功 / 80 | 0 | 0 |

逐 case 数量仅解释已生成程序；训练 reward 始终是整套测试全部通过的 binary 值，没有给予部分通过奖励。Token 开销不同，因此也不能将这里当成等训练预算实验。

| START 首 route | v1 | v2 |
|---|---:|---:|
| start | 63 | 2 |
| NONE | 21 | 0 |
| end | 9 | 0 |
| n1 | 2 | 0 |
| n0 | 1 | 94 |

v1 有 87.5% 入口立即自环或拒绝，v2 为 2.08%。这是本次配对探测中清楚的行为变化，支持先修正通用命名说明与 grammar 的不一致。进入 n0 之后仍有大量语义失败：新 screen 1380 个失败 case 中，867 个最终输出不符、391 个达到步数上限、122 个路由到 NONE。入口改善没有解决 FIFO 变换算法本身。

## append_R 的实际成功程序

文件 `job_41415498/samples/01_append_R_concise.json`，sample index 7（从 0 计数），reward=1，16/16 train case 全通过，每次 4 步。保存的模型程序全文为：

```manufactoria
START start:
    NEXT n0
PULLER_YG n0:
    [Y] n0
    [G] n0
    [EMPTY] n1
PAINTER_RED n1:
    NEXT end
PULLER_RB n2:
    [R] end
    [B] n2
    [EMPTY] n3
PAINTER_BLUE n3:
    NEXT end
END end
```

输入域为 RB。n0 的 PULLER_YG 看不到可移除的 Y/G，会走 EMPTY，且原 tape 保持不变；n1 向队尾追加一个 R，再结束。例如原 train case B→BR、R→RR、BR→BRR，符合任务。n2、n3 从入口不可达。该程序的图上有循环，但 RB 执行不走这些循环；因此它不能被当作最终 target 的长程循环解或循环能力提升证据。

新 identity 成功在 `00_identity_concise.json` sample index 0：START→PULLER_YG n0，其所有 route 到 end。RB 输入走 EMPTY 并保留 tape，16/16 通过。两个成功程序都是模型此次实际采样的文本；审计只在事后解释它们，没有将它们作为训练提示、reference 或 teacher path。

## mixed 组与实际更新不能混算

新 screen 出现 identity、append_R 两个 mixed 组；`probe_prompt_v2.py` 仍按预先存在的 first_mixed 规则只对第一个 identity 组做一次更新。该组 rewards 为 `[1,0,0,0,0,0,0,0]`，822 个 native tokens，proof 记录 822 个 nonzero-advantage tokens、loss `-0.03771668991613272`、同 grammar mask 的 native-suffix policy loss、beta=0、weight_decay=0、parameters_changed=true、optimizer_steps=1。**append_R 的 mixed 组在本 screen 中没有执行 optimizer step**。

远端只读权重哈希确认原 adapter 仍为 `9596df...a5f5a321`，新 proof 为 `56d7f53ba26b053e5c778a3d61343f08b5488adeea48b5fe5e7dc6c57de75e98`；proof 保留 `round_base_changed=false`。这确认了真实一次 binary 参数更新，不代表更新后有更高 stage 成功率；screen 未做更新后的 stage 评价，更未证明 target transfer。

## 成本、复现和下一轮边界

Slurm 记录作业 41415498 COMPLETED、exit 0:0，2026-09-08 14:33:47–14:37:19，212 秒；分配 1 B200、4 CPU、64G，相当于 0.05889 B200 小时。batch step MaxRSS 为 2,799,484K；它不是整体显存峰值。完整原记录见 `naming_v2/sacct_41415498.txt`。

重放命令为 selfgrok release 下 `bash evidence/naming_v2/RUN_REPLAY_CPU.sh`。最终产物为 `cpu_replay_metadata_complete.json`、`cpu_replay_metadata_complete.log`、`adapter_hash_verification.txt`、sacct 记录。首份 `cpu_replay.json` / `.log` 保留；最终版本另外将 metadata 缺失键与 null 区分，全部程序计数及结果不变。审计仅 CPU，未提交任何 GPU 任务或修改原始 screen/checkpoint。

已经冻结的 `benchmarks/generated/naming_round_v2` 保留原 128 train、8 selection、每题 36 tests、六类映射、原研究问题和 stage specs；新增的是同一段通用命名说明及明确的新 prompt-view 身份。其目标模型表现须由新一轮实际 base、curriculum、direct 评价来判定。本报告不读取或预判正在执行的后续 round，也未启用 numeric naming 备选。
