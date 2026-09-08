# 为什么没有出现目标成功和正 Gamma

本报告是对已结束的三轮开发实验的事后诊断。依据为原始设计、实际训练 journal、去重后的 selection 评价和实际程序，不使用 held-out/test 模型结果，不改变正式测试类、奖励、旧报告或 checkpoint。本次只进行 CPU 分析，没有新增 GPU 训练。

首要瓶颈是可用学习信号几乎没有出现；唯一一次更新来自 identity，尚无目标相关能力提高的证据。当前实验验证了训练、评价、评分链路可执行，但没有达到“课程能够带动目标学习”的实验条件。此前以完整运行且出现一次参数更新作为最小流程验收标准，容易低估与用户所要研究成果之间的距离；后续应明确区分流程完成与可学习、可迁移的证据。

## 1. 生成预算大部分没有转化成更新

| 开发轮 | 四分支训练生成 token | 实际 Solver optimizer steps | 三条课程 Gamma |
|---|---:|---:|---|
| 1 | 262,144 | 0 | 全部 0 |
| 2，命名说明修复后 | 262,144 | 0 | 全部 0 |
| 3，g1 首阶段加入已有 mixed 证据的提案先验 | 262,144 | 1 | 全部 0 |

合计 786,432 个训练生成 token，只有一个 924-token 组带有非零 advantage，占 0.1175%。这是处于非零优势组的 token 数，不表示 grammar 强制的每个 token 都有非零梯度。前置 screen 的两次独立更新证明没有接成这些分支的初始权重，不能计作课程累积训练。

逐组复算得到 697 个完整八样本组：696 组全零、0 组全一、1 个 mixed。5,576 个组内程序只有一个 reward=1；另有 44 个预算收尾样本、172 tokens，不计作 optimizer group。目标任务单独占 383 组、3,064 个程序，全部失败。identity 实际采样九组、72 个程序，仅一个成功。完整统计与源文件哈希见 `evidence/result_analysis_training.json`。

这与原文的 Solver 规则一致：全组 binary reward 相同就跳过更新。全零组的优势为零；增加相同分布下的这类采样会花费预算，却不会在这些组上积累训练。日志、vendor 重放和 LoRA tensor 直接比较支持这一解释。不能仅凭一次小更新，将失败归因于学习率不足、模型容量不足或 GPU 数量不足。

来源：三份 `runs/round*_result/budget_audit.json`、各分支 `execution/training_journal.jsonl`、`evidence/ROUND3_UPDATE_REPLAY.md`；实际更新实现为 `runtime/hf_backend.py::Solver.update`。

## 2. 唯一成功没有补上到目标的能力差距

第三轮 g1 是 `identity → target_small3 → prepend_RB`。唯一 reward=1 的模型程序是：

```manufactoria
START start:
    NEXT end
END end
```

它保持输入不变；固定目标则要求 `RBYGRB + input.replace("RB", "YG")`。因此该成功程序不执行目标前缀、替换或 FIFO 循环处理。一次更新是否提高了 identity 的后续成功率，本实验也没有独立的 stage 前后评价可以证明。

`target_small3` 虽然缩短了输入，仍保留完整目标变换，并没有变成已验证可学的中间技能。catalogue 中 identity 参考程序为 2 节点，small3 仍采用 18 节点目标参考程序；这只是参考实现大小，不能当作最短程序复杂度证明。g1 后续阶段及 target-only tail 没有继续产生参数更新。固定预算段结束就切换 stage 是本轮冻结规则，但本次切换前没有形成已测出的技能进步。

Q 的第二轮九个位置都没有选择 screen 中出现过 mixed 的 identity/append_R。第三轮的公开先验只修正了 g1 第一位置，后面两段仍缺少可学性证据。长 context、模型对指令的理解等可能影响提案，但当前没有实验能识别其因果作用，不能当作确定原因。

来源：实际三轮 plan、`benchmarks/README.md`、`benchmarks/stage_catalogue.py`、`evidence/q_ignition_review_20260908/REVIEW.md`、第三轮 g1 原始 mixed 组 `execution/raw_training/000010_g1_segment_1.json`。

## 3. 此次全零不是 min-per-class 掩盖了已有 exact-pass 改善

对三份物理生成的目标评价去除缓存 alias 后重新汇总：

| 物理评价 | 程序数 | 可解析 | 逐测试 exact-pass | 至少通过一个测试的程序 |
|---|---:|---:|---:|---:|
| 第一轮初始 Solver，原 view | 256 | 256 | 0 / 9,216 | 0 |
| 第二轮初始 Solver，命名 view | 256 | 256 | 0 / 9,216 | 0 |
| 第三轮 g1，identity 更新后，同命名 view | 256 | 256 | 0 / 9,216 | 0 |
| 合计 | 768 | 768 | 0 / 27,648 | 0 |

每份数据六类各有 1,536 次 case 执行，全部零通过；所有程序的 f 都为 0。没有把缓存评价重新计为模型采样，也没有把这些固定目标 suite 当作独立任务族。第一与第二份改变了 prompt view，不能归因于训练；第二与第三份比较的是同 view 下原权重与更新后的权重。

可复算产物为 `evidence/analyze_target_eval.py`、`evidence/result_analysis_eval.json`、`evidence/result_analysis_eval.log`。脚本核对原始 generation SHA、逐测试结果、selection 身份、去重以及 view/权重/采样契约，汇总检查 PASS。

原命名版初始评价的失败为输出不符 5,437、超时 2,600、NONE 1,179；更新后为 5,253、2,892、1,071。这些变化没有带来一个目标 case 的正确输出，不足以说明有学习改善或退化。

min-per-class 确实要求每一类至少出现通过测试，才能越过第一条 fraction condition。在当前六类各六个测试的配置下，阈值 1/36 意味着每类至少 1/6 通过，并非整套测试任意一条通过。但本次连整体 exact-pass 都为零，所以换成整体通过率仍然为零；改变这个聚合不是当前证据支持的解法。

语法修复确有作用：screen 可解析率从 1/96 提高到 96/96；命名说明对齐后，首跳 start/NONE 从 84/96 降到 2/96。它解决了输出格式和入口问题，未证明目标变换算法已学会。程序即使部分操作更合理，只要每个测试最终输出仍不正确，现有 exact-pass 条件就不会给出反馈。

同一批保存程序还用原 vendor 进行 CPU 重放，27,648 次 case 执行与原始记录相符。在终止的输出中，三份数据都没有正确的 `RBYGRB` 前缀；在实际需要 RB 替换的输入上，也没有观察到正确的 replacement-only 输出或去掉前六位后正确的 replacement payload。这些只是事后诊断，不是新奖励或正式测试类，也不能证明模型内部完全不具备相关知识。

CPU 重放结果与日志为 `evidence/result_analysis_eval_cpu.json` / `.log`，状态 PASS。原 parser SHA256 为 `fa8bb2a00447c0c0f73f5bec34759d60fea08c5c67e99ff0a06850e421acdb72`。三份数据终止 case 的分母分别为 1,333 / 5,437 / 5,253；其中需要真实替换的终止 case 分母为 888 / 3,639 / 3,540。无需替换时保持输入可能满足 replacement-only，故没有将那类平凡匹配当成已学会替换。

静态图含环的程序数为 184、204、204；实际某次测试路径重访节点的程序为 146、155、153。其中静态含环但所有该批测试都未重访节点的程序分别有 38、49、51。能重访节点并终止也不等于正确循环算法。它解释了为何当前结构格 s=1 很容易被占据，却没有带来正确目标输出。

## 4. 当前反馈全零，但不能据此否定完整方法

当前 rho 为 `[1,0,0,0,0,0]`，frontier 停在第 2 条。课程和 direct 都没有越过它，所以 Gamma=0，J+ 为空，Q 没有正例用于 RFT。降低 J+ 的样本数门槛不能从零个正例制造训练数据。bootstrap 区间 [0,0] 反映本批观察差值全零，不是总体潜在收益严格为零的证明。

因此原文“first success 之前非退化”不能当作无条件保证：逐测试结果比全题 outcome 有潜在更细的信息，但仍可能全部同值，或者其差异尚不足以越过冻结 condition。当前数据正是所有未饱和条件仍为零的情形。这要求验证方法适用的学习区间，而不是假定定义了 frontier 就自然有可用于选择的增益。

原文已经把“目标能否点燃”列为单独的前置问题。本次只证明任务有可执行 oracle 和 identity 能偶尔出现 mixed，未证明目标相邻 stage 可以稳定产生学习并迁移；原 Gate 1 的可学习性证据尚未完成。实际 Q 只从九项菜单选择，hinted target 没有进入课程，开放变换搜索与正例 RFT 也没有形成有效运行证据。不能据此说三类 stage 来源已经全部尝试，或完整 VERGE 已经被否证。

还有一项实现边界：三轮都从原始 Solver 开始；`round/round_cli.py` 每次新建 `CheckpointArchive()` 和 `RejectionSamplingBuffer()`，过往记录进入 Q context，但这不是同一 Solver 种群的持续三轮学习。这个边界必须修复才能检验长期闭环；然而在本次所有候选同格且没有增益的情况下，单独把 archive 持久化也不会自动产生可学 stage，因此它不是本次零梯度的直接原因。静态节点图有环也不代表实际执行学会了有效循环。

## 5. 下一步应回答什么

优先验证“可学且目标相邻的 stage 是否存在”，再花预算运行下一组完整课程。研究问题、binary Solver reward、固定目标无 hint 评价、原本 P1–P7 对照范围保持不变。

1. 对候选中间任务做有界可学性实验，测真实 mixed 频率、实际多次更新以及独立的 stage 前后成功率。偶然一个 identity 成功不再作为目标可点燃性的证据；CPU oracle 通过也不替代模型测量。
2. 优先使用原方法允许的注册/自写目标相邻变换和可追溯的已验证程序库 hinted target。需要 hint 时按原两臂 512-rollout 协议验证 hint-regret，并测撤掉 hint 后的目标表现；不能把事后诊断、参考解或 dense reward 静默接入 Solver。
3. 找到可学入口后，保持共享 base、匹配实际生成 token、direct 对照和无 hint 评价，再检验课程迁移。阶段预算、候选空间若调整，预先版本化；不在正在运行的分支中临时延长有利阶段。
4. 若这些原方法来源在当前任务/模型组合下仍无法产生可迁移训练，再依据用户授权更换 benchmark 的具体实例或难度；仍需保留固定 target、binary 训练、课程/checkpoint 搜索及全部科学对照，重新核实 loop/direct regime，不能把 identity 成功改称目标完成。

应期待的是能判定下一步的证据；不能承诺一定得到正 Gamma。当前最直接的改进是补齐学习与迁移之间的验证，重复同样的全零课程不是有根据的下一步。

## 本次分析的复算与保存

新增的训练汇总脚本为 `evidence/analyze_training_results.py`；其复算与原诊断 JSON 逐字节一致，日志为 `evidence/analyze_training_results_CPU.log`。主 agent 另行执行得到 `evidence/result_analysis_training_rootcheck.json` 与 `.log`，退出码 0。目标评价诊断见上面的 `analyze_target_eval.py` 及 JSON/log。

以下命令以本 release 为当前目录，输出文件名须未存在，`python` 为可用的 CPU Python：

```text
python -B evidence/analyze_training_results.py --output evidence/result_analysis_training_RECHECK_NEW.json
python -B evidence/analyze_target_eval.py --output evidence/result_analysis_eval_RECHECK_NEW.json
python -B runtime/verify_delivery.py
```

最后一项本轮执行 PASS：17 个冻结交付文件（含训练后 LoRA 权重）仍与原 manifest 完全相同。日志为 `evidence/RESULT_ANALYSIS_DELIVERY_CHECK.log`。新增本报告和诊断产物独立保存，不更改此前的实验结果。
