# VERGE 可执行 benchmark 适配，2026-09-08

本轮依据用户“可以改变 benchmark，但研究问题和所有 scope 不变”的授权，新增 `verge_prepend_rb_v1_20260908`。它保留现有 target 的输出函数 `RBYGRB + x.replace('RB','YG')`，把**输入域明确限制为 R/B，长度 0–64**；Manufactoria DSL 仍提供 R/B/Y/G 四符号与原有所有节点。原四色 benchmark、代码、checkpoint、结果与作业没有被覆盖。本文件是本轮新增操作化定义，不声称恢复了旧稿未找到的 class 定义。

新 benchmark 已有原解释器实测通过的 18 节点循环解、冻结分组与数据生成器。**目前证明的是任务可解且工程可执行，没有证明当前 Solver 处于零/稀疏 full-pass regime，或自动课程优于 direct。** 这些仍按原 P1–P7 范围检验，不会把 CPU oracle 当模型实验。

## 原依据与变更

- 原方法：`../../verge_review_20260908/evidence/VERGE_own_method_design.md` §1.1–1.6、§4。
- 现有 target 定义：同审计目录 `review_copy/src/verge_target_data.py:PARAMETERS`，前缀 RBYGRB，替换 RB→YG。
- 原执行器：`/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py`，SHA256 `fa8bb2a00447c0c0f73f5bec34759d60fea08c5c67e99ff0a06850e421acdb72`。本轮使用原文件，没有改变 1000 步、puller 的 EMPTY 路由语义或节点语法。
- 变更仅涉及 benchmark 输入域、有限测试支持集、冻结分组和 split。原 task family、前缀、mutation 规则均保留。目标在本轮版本冻结后不能在一个正式对照实验途中换题。
- 原四色问题不等同于新输入域：当前参考解对旧输入 `Y` 输出 `YRBYGRB`，不等于原期望 `RBYGRBY`。这个反例已实测保存，避免把两套结果混用。

## 循环解为何成立

参考程序在队尾追加 Y，再追加固定前缀，得到 `x Y RBYGRB`。由于输入只含 R/B，前方第一次遇到的 Y 一定是这个边界；程序以“正常 / 暂存一个 R”两种状态扫描原输入，把 `RB` 输出为 `YG`，其他字符原样输出到队尾。此时新输出位于前缀之后，不会被误作原输入再次处理。消费边界 Y 后结束；若仍暂存 R，则补写 R 再结束。

这一不变式给出最终 tape `RBYGRB + replace(x,RB,YG)`。每个原输入字符各需一次 pull 和一次 paint，加 11 个固定执行步骤，总步数 `2|x|+11`，域上最大 139，小于原解释器 1000 步上限。CPU 实测与该上界一致。

参考 DSL 仅是 oracle、可解性见证和 CPU 工程 fixture。生成的数据只有任务 prompt、输入和期望输出；不包含参考程序、路径、动作、logits、人工分解或 imitation loss。最终 target prompt 无 hint。此程序不作为 Solver 训练监督，也不自动加入真实训练的 hint 程序库；后续 `integration_round.py` 只在明确标注的 synthetic CPU fixture 中用它检验临时程序库和 hint prompt 接口。

## 冻结的各测试类

`benchmark.py:test_class` 按字符串中非重叠 RB 的数目与输入是否以 R 结束分组。前者覆盖“从未触发、触发一次、重复触发替换”路径，后者覆盖边界到达时 transducer 是否仍暂存 R。这是目标自动机的语义覆盖，不依据模型输出或 held-out/test 模型表现。六类互斥且覆盖全部域；不声称六类统计独立。

| 类名 | 归属规则 | 示例 |
|---|---|---|
| matches_0__pending_r_0 | 没有 RB，非 R 结尾 | 空串、B、BBB |
| matches_0__pending_r_1 | 没有 RB，以 R 结尾 | R、BBRR |
| matches_1__pending_r_0 | 恰一次 RB，非 R 结尾 | RB、RRBB |
| matches_1__pending_r_1 | 恰一次 RB，以 R 结尾 | RBR |
| matches_2plus__pending_r_0 | 至少两次 RB，非 R 结尾 | RBRB |
| matches_2plus__pending_r_1 | 至少两次 RB，以 R 结尾 | RBRBR |

每个 suite 固定每类 **6 条，共 36 条**；每类分母恒为 6，timeout、NONE、执行错误和错误输出各计该测试 0。任何空类、漏项、重复输入或被改写的输出/分组元数据均拒绝；全类最小 exact-pass fraction 为 f，Solver reward 仅为全部 36 条成功的 0/1。

最初 24-test 草稿每类 4 条，导致原条件阈值 `1/24` 与 `.25` 完全同事件。本轮审查发现后保留全部草稿到 `generated_24_draft/`，改为每类 6 条，**不改原方法阈值**。正式阈值 `{1/36,.25,.5,.75}` 分别需要每类至少 `{1,2,3,5}` 成功，四个事件区分明确。测试覆盖了这项性质；它只消除了结构上的重复条件，不保证模型实际取得非零 f。

## 固定 split 与冻结 artifact

`benchmark.py:split_of` 仅用 `SHA256(version+'|input|'+tape) mod100`：0–49 train，50–69 selection，70–84 heldout，85–99 test。类和 split 在任何模型采样前确定。输入支持集为：长度 0–12 的全部 RB 串；长度不超过 64 的全部 `B^aR^b`；固定 seed 的 4096 条长度 13–64 随机串；重复 RB 路径，去重后 14,417 条。每个 suite 按固定 seed 在所属 split 每类采 6 条；同 split 可以复用输入，完整 suite 不重复。

| split | suites | 每 suite 测试 | 实际唯一输入 |
|---|---:|---:|---:|
| train | 128 | 36 | 2396 |
| selection | 64 | 36 | 1087 |
| heldout | 64 | 36 | 987 |
| test | 64 | 36 | 1016 |

六对 split 的输入交集全部为 0。空串属于 `matches_0__pending_r_0`，hash 将其分到 selection，并显式保证该 split 首个 suite 覆盖它。没有为了让训练见到空串而重分配。每个 split 的六个池都非空且至少 6 条；最小池 heldout/no-match/non-R-tail 有 7 条。

不相交的是 `(benchmark_version,input_tape)` 和 suite；固定目标的任务参数在所有 split 相同。同一 split 的 suite 复用一些输入且 prompt 相同，不能把 suite 当独立任务族或把重复 oracle 执行当独立模型训练 seed。本轮生成 heldout/test 输入和 oracle期望并作解释器校验，没有读取或生成这些 split 的模型结果。仅有固定输入分离还不构成官方密封 benchmark。

`generated/manifest.json` 记录支持集、prompt、wrapper、各 split 文件 SHA256 和分母。Windows/Linux 写出使用固定 UTF-8/LF；`local_remote_consistency.log` 证明本地与远端所有 manifest 字段及各数据文件 SHA 一致。

## 实际 CPU 验收

- 原 parser：长度 0–14 全枚举 32,767 条，全部精确通过。
- 额外长串与语义覆盖：5,998 条，长度到 64，全部精确通过。
- 正式 320 个 suite 共 11,520 次 oracle 执行，全部精确通过。
- 前两项共 38,765 个不同输入，其中 38,763 条执行实际重访节点；最大执行步数 139。静态含环与实际执行循环都有证据，但它们不是原 P6 的模型因果结论。
- 错前缀 negative control 在 7 条预先指定输入上全部失败；域外旧四色输入 Y 的反例也失败。
- 远端原环境 10/10 单元测试通过，无跳过。包括固定分母、f=min而不是整体通过率、缺失类拒绝、parse失败、单测试timeout/exception后继续、原解释器1000步timeout、split不交、目标prompt无oracle、四个fraction条件互异。
- 本地 Python 缺 networkx/matplotlib，因此本地 9项通过、1项真实parser测试跳过；完整验收使用 self grok 既有环境，没有安装新库。最初 WSL sandbox拒绝/缺库与 scp空格转义失败只影响环境接入，后续成功命令见日志；未隐藏算法测试失败。

证据：`cpu_reference_verification.json`、`cpu_tests.log`、`cpu_verifier.log`、`local_remote_consistency.log`。`RUN_CPU.sh` 给出可复制的实际 CPU 运行口径。

## 研究问题与 scope 保持

仍研究固定 target 上零/稀疏 full-pass 条件下，Challenger 能否通过自动课程搜索帮助 Solver。仍保留 registered / 自写 tape transform / hinted target 三来源、无hint最终target评估、Solver binary full-pass、fresh optimizer与同奖励组跳过、同base compute-matched direct、(frontier,structure)多格archive、J+及RFT、原P1–P7预测和反证标准。此次文件实现 benchmark/verifier 层，不声称把这些组件都实现了；集成状态由主交付审计报告负责。

不引入 CAFD、T1/T2、教师路径蒸馏、KD/RL路由或MPC。原“首次成功前非退化”和 hint迁移不能因为有手写oracle就宣布成立。有限最大长度允许理论上构造很大的无环unroll程序；存在18节点循环解不等于数学上所有解必须有环，更不等于 P6 的 matched-compute direct/dense 对照结论。

仍需在此冻结版本用选定真实 backbone/checkpoint 做最小有预算的 regime测量，然后检验stage的binary reward混合组、无hint迁移、完整自动课程以及原对照。本 benchmark 与 CPU 集成子任务没有运行模型采样、GPU smoke、训练或Gate，没有提交Slurm任务；主任务若另行执行真实模型 smoke，由主审计报告单列。资源上限仍是用户所限4 B200、192GB；上限不替代具体预算授权。

## 接口

`benchmark.evaluate_program(program,cases,create_factory)` 返回 `parse_valid, per_test, class_rates, f, reward`。`create_factory` 传入上述冻结原 parser 的 `create_robot_factory`。它要求 `ground_truth` 中的 `test_class` 保留。**旧 `TrainingFileWrapper._extract_test_cases` 会丢弃这个字段**，不能把新数据重新经过该旧提取函数；本生成器仅复用旧 prompt 模板，自行写出带完整分组元数据的 rows。

正式训练仍需使用主任务的新方法入口，不能直接把这些文件塞入旧11维ladder/旧loss-token预算入口并宣称新版运行。
