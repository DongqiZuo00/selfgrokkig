# Canonical 命名与提示对齐复核

2026-09-08。只读核查已有 grammar、真实 tokenizer、catalogue 和已下载的 train-side 原始采样；未改冻结文件、未加载模型、未运行 GPU、未新增 validator。逐文件 SHA、计数和 tokenizer 证据见 `PROMPT_ALIGNMENT_EVIDENCE.json`。

**结论：工程假设的前半部分成立，值得做只改命名说明的 bounded probe。** 现有提示没有告诉模型必须用 `n0,n1,...`；grammar 却删除了其他名字。已有样本显示，命名偏好主要是数字 ID，也包括语义 ID。grammar 下出现大量 `START → start/NONE` 是事实，但“命名不匹配导致这些首跳”仍是待检验的因果解释，不能当成已经证明的根因。

## 1. 提示与 grammar 的确定差异

- `benchmarks/stage_catalogue.py:139–162` 的 `COMPACT_SYNTAX` 使用 `<node_id>`，只要求 unique IDs；没有 `n0`、连续编号、声明顺序、最多 30 个中间节点这些约定。`prompt()` 的 official 版本还允许字母数字/下划线 ID，并给出 `entry` 示例，同样没有 canonical 约定。
- `decoding/dsl_grammar.py:26–39` 对 K 个中间节点只允许声明 `n0` 到 `n(K-1)`，无间隔、按编号递增；K=0…30，另含固定 START/END，总计 2…32 个节点。每个 route 只能是 `start`、`end`、`NONE` 或该完整程序内声明的 `n...`。START 首跳未被强制为 n0，前向/后向引用和环均保留。
- `benchmarks/freeze_pilot.py:80` 把同一个 `COMPACT_SYNTAX` 用于当前 target view。已核查的 24 组原始记录中，实际 prompt 均不含 `n0`。这是显式语法说明的缺失，不是任务目标需要改变。

grammar 还固定了 route 顺序和排版，official wrapper 的自由评论/空行等说明比实际 grammar 更宽。本次最小 v2 仅隔离 canonical 命名说明这一变化，不顺手更改 grammar、任务文字或全部排版。

## 2. 已保存模型采样的直接证据

两个 stage screen 均为 12 组 × 8 条，使用相同初始 Solver。复核逐组匹配实际 prompt SHA、seed 和 cap；既有 `evidence/stage_ignition/raw_audit.json` 另记录相同初始 adapter digest。以下首跳来自原始 `verifier_completions` 的 START route；无约束输出多不合法，因此称为输出中的首跳拼写，不将其视为可执行图。

| 观察 | 无约束 41410864 | Grammar 41412533 |
|---|---:|---:|
| 实际样本数 | 96 | 96 |
| Parse-valid | 1 | 96 |
| Binary full-pass | 0 | 1 |
| 首跳 `start` | 0 | 63 |
| 首跳 `NONE` | 0 | 21 |
| 首跳 `end` | 0 | 9 |
| 首跳 `n0` / `n1` | 0 / 0 | 1 / 2 |
| 首跳恰为 `0` / `1` | 41 / 30 | 0 / 0 |
| 出现非 canonical 中间节点声明拼写的样本 | 68 | 0 |
| 实际生成 tokens，含 EOS | 62,977 | 10,089 |

Grammar 的 84/96 条首跳直接为 `start` 或 `NONE`；其余节点即使存在也无法绕过这个首跳。48/96 条没有中间节点。唯一成功样本位于 `job_41412533/samples/00_identity_concise.json` 的零起算 slot 7，采用 START 直接到 END，满足 identity。保存的 `one_binary_update_proof.json` 确认该真实 mixed group 完成了一个 masked LoRA update；不能从 identity 的成功推出非 identity stage 已经学会。

无约束样本中，除数字 ID 外，首跳还包括 `painter_red`、`painter_blue`、`fillLastSymbol`、`paint_room_R`、`tape_painter`、`pull_first` 等。它们说明模型会提出 descriptive ID；但只用“模型想写 paint_red/process”概括不够准确，因为数字命名更占优势，且许多输出同时存在其他 DSL 错误。

根任务报告当前两个 branch 的第一阶段各用了 16K tokens、均为零成功。本次本地未发现这些 branch 对应的下载 raw，故只把它登记为运行状态，不将其加入以上独立计数，也不读取任何 held-out/test 模型结果。

## 3. Tokenizer 和 mask 层面的解释

只读实际 `/blue/du.j/jinjiaguo/self grok/models/Ministral-3-3B-Instruct-2512-BF16/tokenizer.json`，其 SHA 为 `d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135`，与之前跨环境 replay 检查一致。实际 BPE 词表中：1048=`0`、1049=`1`、1110=`n`、10460=`start`；`start`、`NONE`、`end` 也存在多种分词路径。

无约束样本的第一个 native token 有 45 条为 1048、35 条为 1049，共 **80/96**。这比首跳恰为 `0/1` 的 71 条多，因为还含 `0x00`、`100...` 等延续。固定 prefill 已结束于 `NEXT `，grammar 当前位置只能开始 `start/end/NONE/n...`，所以这两个 token 不能作为第一个生成动作。Grammar 样本中只有 **3/96** 条以 1110=`n` 开始。

这不是 tokenizer ID 错位。已有真实 tokenizer CPU replay 证明两环境 vocab 一致、生成 mask 与训练 replay 相同。这里的问题是模型被提示允许的命名空间比实际允许空间更宽。

Mask 没有“失败后选 start”的回退逻辑；它把禁用 token 的概率置零，再在允许 token 内重新归一化。若模型在允许集合内原本给 `n...` 的概率低，其他合法首跳便可能占主导。**现有 raw 没保存未 mask logits，因此无法量化被删除的概率质量，或确认该解释足以覆盖当前失败。** 加上提示对齐后，也可能仍有任务理解、图结构或探索不足；不能预告 v2 必定得到正奖励。

## 4. 最小 PROMPT v2 文本

将下面同一段 canonical 格式说明 append 到既有 user message；保留原始 task、tests、instance 身份、grammar、prefill、checkpoint、sampling 参数和奖励。它只是把正在执行的命名约定告诉模型，不附示例程序或解法：

> Use this canonical naming format. Keep START start first and END end last. If there are K intermediate nodes (0 <= K <= 30), name and declare them n0, n1, ..., n(K-1) in that exact order, without gaps or other node names. In node declarations, <node_id> is the next canonical name. In routes, <node_id> may be start, end, NONE, or any intermediate name declared in the program, including one declared later. Declaration order does not determine execution order. Use four-space route indentation. END has no colon.

这段不规定 START 指向哪个节点，不禁止 `start`、`NONE` 或自环，不建议中间节点类型、数量或任务拓扑。`n(K-1)` 是命名说明中的数学记法，与前文 angle-bracket placeholders 一样不是要求输出的字面 ID。

## 5. 下一次 bounded probe 的可判读范围

可直接用上述单一 prompt delta，在同一 untouched initial Solver 和同一 grammar 下重跑原 12 组 × 8 条、相同 seed/cap。2,048 cap 下最多 196,608 native tokens，实际成本按新 raw 计数；不因为旧 screen 更短就承诺相同耗时。若需要 update proof，将其放在 screen 完成之后并单独记账，避免前组更新污染后组比较。

主要记录仍为原严格 verifier 的 binary full-pass、parse-valid、mixed groups。START 首跳分布、生成长度和 n 名称使用率只是诊断项，不进入 reward，也不成为新 validator。若有变化，只能支持这次 prompt 对齐的工程效果；若仍零非 identity 成功，也保留完整 raw，不改成强制首跳或拓扑门槛来制造正例。

如果 v2 后续进入正式四分支比较，需要新建冻结 prompt view，并为该 view 重新测量其 baseline；不可拿 v1 baseline 和 v2 branch 直接计算 Gamma。当前 v1 运行、日志和 checkpoint 全部保留。
