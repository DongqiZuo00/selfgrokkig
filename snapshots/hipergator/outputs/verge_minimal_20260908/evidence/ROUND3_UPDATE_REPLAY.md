# 第三轮 g1：真实 mixed 更新与 checkpoint 独立复验

第三轮 g1 已在 identity stage 出现一个真实 mixed-reward 组，并完成一次 binary-only Solver 更新。原 vendor CPU 重放该组全部 8 条输出和紧随其后的 8 条 constant 组，**16 条程序、416 次 case 执行全部复现保存结果**。独立 CPU 比较 base 与阶段 checkpoint 的全部 364 个 LoRA tensor，确认 182 个 tensor、12,779,520 个元素实际发生变化；这不是只依据日志或文件哈希推断的更新。

该证据证明真实 stage mixed 奖励能进入训练并改变 Solver 参数。它不证明 identity 训练改善了最终 target，也不代表第三轮完整 curriculum/direct 对照结果。

## 固定证据范围

原始运行是 `runs/round3_g1_41420288_0/execution`。只读快照保存在 `evidence/round3_update/snapshot_first_update`：启动契约、计划、training journal、首个 mixed raw、随后一个 constant raw。原 checkpoint 保留在运行目录，没有下载完整权重。

快照共 66 条 journal 记录，hash 链逐条验证通过。g1 第一阶段已经完成 16,384 generated tokens，113 个 rollouts，1 次 optimizer step，13 个 constant 组，1 token boundary drain；其中 924 tokens 对应 nonzero advantage。阶段 checkpoint 已落盘。快照末尾仅多出下一阶段开始的记录，本审计不读取 target evaluation 文件、held-out/test 模型结果或其他 branch 结果。

启动计划、catalogue、target rows 和 runtime contract 中全部 20 项源文件哈希均重新核实。原始 contract 字节哈希与运行目录 JSON 副本内容分别比较，保留换行序列化差异的事实。g1 第一阶段确实是实际 Challenger 计划中的 identity，使用原命名对齐的 concise stage row；并未在审计时替换课程。

## 首个 mixed 组

原 raw 为 `raw_training/000010_g1_segment_1.json`，对应 generation_source sequence 42、rollouts sequence 43、actual_model_update sequence 44、accounting sequence 45。

- instance：`verge_ignition_stage_catalogue_v1_20260908__identity__compact`。
- 8 条 raw rewards：`[0, 0, 1, 0, 0, 0, 0, 0]`。
- 实际生成 924 个 native tokens，组 cap 562；optimizer_steps_before=0。
- 8/8 程序可解析；sample index 2 的 16 个 train case 全通过，其余 7 条各自未完整通过。

成功样本的实际程序为：

```manufactoria
START start:
    NEXT end
END end
```

它保持每条输入 tape，符合 identity stage。这是本次模型采样出的 stage 解，没有作为 reference 或 teacher path 输入训练。该程序不完成最终 target 的 prepend/mutation 变换。

实际更新 metric 记录：tokens_used=924、nonzero_advantage_tokens=924、optimizer_step=true、parameters_changed=true、optimizer_steps=1、loss=`-0.03896356725589537`、loss_policy=`same_dsl_grammar_masked_native_suffix`、beta=0、weight_decay=0、binary_reward_only=true。审计核对 metric 的 raw 路径、native token 数、reward 与 journal rollout 完全一致。

| Ledger 字段 | mixed 前 | mixed 后 | 变化 |
|---|---:|---:|---:|
| generated_tokens | 11,888 | 12,812 | +924 |
| loss_eligible_tokens | 11,888 | 12,812 | +924 |
| nonzero_advantage_tokens | 0 | 924 | +924 |
| optimizer_steps | 0 | 1 | +1 |
| constant_groups | 10 | 10 | 0 |

原 vendor parser SHA256 为 `fa8bb2a00447c0c0f73f5bec34759d60fea08c5c67e99ff0a06850e421acdb72`。该组 128 次 case 执行的 parse、binary、输入/test ID/pass/路径步数全部复现。未通过的 112 次 case 为 96 次输出不符、16 次步数上限；96 个输出不符诊断沿用已登记的 `output_mismatch` / `exact_output_mismatch` 名称映射，不改变 reward。

## 随后的 constant 组确实跳过更新

原 raw 为 `raw_training/000011_g1_segment_1.json`，generation_source sequence 47、rollouts sequence 48、accounting sequence 49。它由同一阶段的既有混合采样规则选中 **target train_0107**，不是 selection 或 held-out/test 评价。

该组 rewards 为八个 0，生成 2078 tokens，optimizer_steps_before=1。CPU 用原 target verifier 完整重放 8×36=288 次执行，返回的结构化结果与原记录完全相等。7 条可解析；另一条在实际 446-token cap 截断，缺少完整 END，按原提取/解析规则失败，未补写代码。没有一条 full-pass。

| Ledger 字段 | constant 前 | constant 后 | 变化 |
|---|---:|---:|---:|
| generated_tokens | 12,812 | 14,890 | +2078 |
| loss_eligible_tokens | 12,812 | 14,890 | +2078 |
| nonzero_advantage_tokens | 924 | 924 | 0 |
| optimizer_steps | 1 | 1 | 0 |
| constant_groups | 10 | 11 | +1 |

这段 source→accounting 之间没有 actual_model_update 事件，冻结实现也只在 binary advantages 非零时调用 update。这里确认的是 journal、计数和代码路径的更新跳过；运行没有保存该 constant 组独立的前后参数快照，因此不声称做过该组前后 tensor 的直接比较。

## 阶段 checkpoint 的实际 tensor 变化

阶段末仅记录上述一次 actual_model_update。`checkpoint_boundary` sequence 63 保存 `g1_m1`，optimizer_steps=1，路径为 `runs/round3_g1_41420288_0/execution/checkpoints/g1_m1/adapter_model.safetensors`。

| 项目 | 独立只读核验结果 |
|---|---|
| 原 base adapter SHA | `9596df4348ee3d959661bf451864f2b8dafe454212bd16928a6774eba5f5a321` |
| g1_m1 adapter SHA | `b5278b1fed5e784945b6770b9abfd3b8d0cede946d2d98ffd771871f2fc86090` |
| 对应 tensor key / shape / dtype | 364 项全部相同 |
| 值发生变化的 tensor | 182，全部为 lora_B；lora_A 变化数为 0 |
| 值发生变化的元素 | 12,779,520 |
| 最大绝对差 | `9.99999338091584e-06` |
| finite 检查 | 原 / 新全部通过 |

比较使用 `safetensors.safe_open(..., framework="pt", device="cpu")`，逐 tensor 调用精确 equality 与差值检查；没有构建完整模型、没有模型推理或 GPU 工作。原 base 文件哈希仍等于开始时的冻结值。独立 tensor 差异排除了仅因 checkpoint 序列化或 metadata 改变而出现不同文件 SHA 的解释。

## 复现文件

在 selfgrok release 执行 `bash evidence/round3_update/RUN_REPLAY_CPU.sh`。`capture_key_evidence.py` 只读 g1 training journal，`replay_update.py` 验证 raw/journal/plan、两组原 vendor 输出与 LoRA tensor 差异。完整结果与每个 tensor 的统计在 `round3_update/cpu_replay.json`，成功日志在 `cpu_replay.log`；原始关键副本在 `snapshot_first_update`。

审计没有提交 GPU、修改冻结文件、改写原 checkpoint 或训练日志。它没有将 identity 正样本、一次 Solver 更新或 checkpoint 变化当作最终 target 迁移证据。
