# 可点燃性 stage 目录：真实模型筛查入口

已实现并冻结 9 个简单任务、两种通用 prompt 写法。原 Manufactoria parser 上，**3,599 条全域 oracle 输入与 130 条训练 suite 测试全部精确通过；11 项 CPU 测试全部通过**。这是任务定义和参考程序的可执行性证据；模型是否能产生混合 binary reward 组、训练能否迁移到固定 target，仍须实际筛查。

本目录属于新 `verge_minimal_20260908`，没有修改旧 `verge_operational_20260908`。目标仍是 RB 输入上的 `RBYGRB + input.replace('RB','YG')`，最终 target 和其六类/36-test 协议保持原冻结版本。本目录提供课程候选，不更换研究问题、不缩减三类 stage 的总搜索空间，也不把手写参考路径作为 Solver 监督。

| stage_id | 定义 | 训练 tests | CPU 全枚举 | 参考节点数 |
|---|---|---:|---:|---:|
| identity | 保持输入不变，长度0–8 | 16 | 511 | 2 |
| append_R | 队尾追加R，长度0–8 | 16 | 511 | 3 |
| append_B | 队尾追加B，长度0–8 | 16 | 511 | 3 |
| replace_R_to_B | 所有R替换成B，长度0–8 | 16 | 511 | 6 |
| replace_RB_to_YG | 非重叠RB替换成YG，长度0–8 | 16 | 511 | 12 |
| prepend_R | 在输入前加R，长度0–8 | 16 | 511 | 8 |
| prepend_RB | 在输入前加RB，长度0–8 | 16 | 511 | 9 |
| target_small2 | 原target函数，输入长度0–2 | 5 | 7 | 18 |
| target_small3 | 原target函数，输入长度0–3 | 13 | 15 | 18 |

每个 stage 在冻结训练 suite 上至少有两种期望输出；最大参考执行步数为23。语法、四种颜色、FIFO pull/paint、NONE和1000步执行上限均来自原 parser，没有更换解释器。参考 oracle 的整域穷举包括空串，是 CPU 正确性校验，不是模型 held-out 结果。

`identity`、append、单字符替换与prepend表达为既有 bounded `tape_transform` DSL；`replace_RB_to_YG` 与两个较小target通过新目录内明确注册的 trusted family 提供，保持原 `str.replace` 非重叠语义。没有放宽旧受限 DSL 的双字符 replace 限制，也没有运行 proposer 自带 Python。`stage_spec()` 与 `registered_families()` 可交给既有四项 validator；这里均为非hint stage，hint-regret明确为NA。后续 hinted-target须通过其独立探针，不能拿这些CPU oracle通过率代替。

## 划分与实例身份

全部训练输入只来自已冻结 RB benchmark 的 `split_of(input)=='train'` 一侧。输入长度≤2时，原hash把空串和RB划入selection，因此 `target_small2` 的训练输入固定为 `B,R,BB,BR,RR`。这份小suite不包含实际RB替换，已在manifest显式记录；另提供 `target_small3`，其训练输入包括RBB/RBR/RRB，覆盖mutation。没有为课程更改原split或借用selection样例。

每个 row 的 `instance_tuple` 由既有 `instance_tuple_from_manifest(task_definition,suite_inputs)` 构造，恰为两个SHA256：真实任务定义与排序去重后的输入集合；不使用split名、row id或prompt版本制造不相交。两种措辞共用同一任务、输入/输出、instance tuple。所有课程输入还与实际selection/heldout/test的输入集合核对，交集全部为0。

这里的小stage定义包含各自输入范围；它们不是独立测试任务族。每个候选目前是一份冻结suite，一组8条生成只是8次程序采样，不等于8个独立训练seed。

## Prompt与模型接口

`concise`（内部别名compact）给出短任务说明和通用节点语法；`official`（内部别名official_syntax）复用原官方wrapper的DSL说明部分，替换task文字。两者都只描述输入、目标输出函数与通用语法，不展示该stage参考解，不提供目标的marker或loop解题步骤。

直接可用：

```python
import stage_catalogue as cat

rows = cat.make_stage_rows("append_R", prompt_variant="concise")
row = rows[0]  # id, messages, ground_truth, task_definition, instance_tuple
# 用 row['messages'] 生成8个独立completion，保持实际completion原文。
verified = [cat.evaluate_stage(raw, row, create_robot_factory) for raw in completions]
rewards = [v["reward"] for v in verified]  # 严格0/1，全部cases通过才为1
```

`make_stage_rows` 返回单个stage的一项row列表；当前可用于每stage×每prompt写法×8条的筛查。`evaluate_stage` 使用已修复的原项目 `extract_program`，不补prefix或END，不按reward选择代码块。它保留每测试slot，单例timeout/exception不抹掉其他测试，并检查期望输出、任务定义和suite身份。

生成文件：

- `generated/catalogue.json`：`stages[{stage_id,kind,spec,rows,rows_by_variant:{concise,official}}]`。
- `generated/screening_rows_compact.jsonl`、`generated/screening_rows_official_syntax.jsonl`：模型行。
- `generated/catalogue_manifest.json`：任务、case名单、两SHA实例键、类别覆盖、源文件与数据hash。
- `generated/references/`：仅CPU oracle。模型rows/messages和catalogue都不包含oracle正文。

筛查应记录8条的全0/全1/混合组以及parse率。全1说明当前已掌握该suite，但该组不会产生binary GRPO更新；全0也不能单独证明不可学。模型阶段奖励仍是当前prompt任务的full pass，不能把本目录的逐测试诊断送进Solver loss。候选是否进入正式课程由真实测量和既有方法决定。

## 实际执行与证据

本地生成仅需标准库及同工作区既有Python模块，不需要vendor、torch或GPU。真实parser验证使用 self grok 既有环境，CPU执行；没有提交GPU任务。

```bash
module load python/3.11
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/mpl"
TASK_PYTHON='/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python'
TASK_PARSER='/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py'
"$TASK_PYTHON" stage_catalogue.py --parser "$TASK_PARSER"
"$TASK_PYTHON" test_stage_catalogue.py --parser "$TASK_PARSER"
```

`cpu_tests.log`记录11/11测试通过，`cpu_oracle.log`与`generated/cpu_oracle_verification.json`记录逐测试oracle结果。覆盖双prompt任务一致、非constant、数据分区、两SHA实例键、small2局限、原target不变、oracle不进入prompt、四项validator、错误元数据拒绝、逐测试异常，以及原parser正负程序执行。
