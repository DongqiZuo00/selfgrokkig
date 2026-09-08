# 实现审计：本轮落地版本

真实仓库是 `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness`。本轮再次核验 HEAD 为 `b3298a61360f659b2a336decbd392252eb429e7c`，原 `common.py`、`verge_round_core.py`、`verge_repair_training.py` 的 SHA256 均与上轮一致。新实现放在其 `outputs/verge_operational_20260908`，本地同名目录保留副本。再次查看 git status 时另观察到 `scripts/recipe_env.sh` 已修改；本轮没有写它，不推定修改者，独立 GPU 脚本也没有 source 它。详情见 `runs/original_engineering_preservation.json`。

## 原文、历史实现、新实现分别记账

| 组件 | 原工程事实 | 本目录实现与依据 |
|---|---|---|
| target | `src/verge_target_data.py` 的 prepend + RB→YG、四色输入 | `benchmark/benchmark.py` 保留输出函数，新增 RB 输入域、六个显式语义类 |
| verifier | vendor Manufactoria parser 有静态 DSL 与 1000 步执行上限；旧 wrapper 会丢掉 class 元数据 | `evaluate_program` 直接保留 case/class；超时按测试失败，其余测试继续；Solver 只取 reward 0/1 |
| conditions | 旧 `verge_round_core.py` 有 11 条件、LCP、整体 fraction | `protocol/verge_protocol.py::TestClasses` 实现最新六条件；显式 expected_classes 防止整类失踪 |
| branches/B | 旧 repair 用 loss-token 配额，实际生成更多 | `plan_round` 规划共享 base 和阶段边界；`runtime/generated_budget.py` 按实际生成计 B、明确 drain |
| Solver | 审计隔离副本已修复 warm AdamW 同奖组移动 | `runtime/solver_update.py` 复制相同函数，AST/SHA 溯源；新增 strict binary/fresh branch 入口 |
| Gamma/J+ | 旧 reward/selection switch 条件不一致 | `score_round` 至少两个分支达到 q；同 kappa 与 direct 配对比较，J+ 要求正增益、正 CI 下界 |
| Delta | 原工程没有可证实的设计定义 | `decompose_stages` 为本轮新增：同 checkpoint/token 边界、固定 kappa、含 tail、显式 32/8 sampling correction |
| evidence | 旧 final 为 64×8，不是最新的 64×32 | `protocol/round_evidence.py` 从逐 rollout 记录聚合；first8 直接引用 final 原始记录，拒绝缺失/重复/错版本 |
| archive | 旧 book 为单血统加历史 JSON | `CheckpointArchive` 真正 `(b,s)` 分格、paired 显著替换、原 lead 规则；零 parse 与饱和单独约定 |
| stages | 旧有注册族和 finite_map，无正式 hint | `stages/operational_stages.py` 的三种 source、有限字符串 DSL、四项 validator 与 hint 程序库 |
| Challenger | 旧在线策略梯度，不是最新 RFT | `RejectionSamplingBuffer` 只收 J+、累计跨 seed、N_min 触发一遍 RFT plan；失败保留新增计数 |
| neural runtime | 旧入口仍执行旧方法 | `runtime/live_smoke.py` 加载实际初始 Solver，作限定规模 binary 链路验证；不把旧正式入口改名视为已迁移 |

`protocol/README.md`、`stages/README.md`、`runtime/TORCH_SOLVER_TESTS.md` 和 benchmark 说明给出接口与行级入口。新增版本不恢复旧工程任何教师蒸馏目标。

## 已完成的实际测试

- 协议 core、raw evidence 与实际格式前缀：36 项 CPU 测试。
- Stage/validator/hint/RFT buffer：18 项 CPU 测试。
- 实际 generated-token 预算、错误日志与 fresh-ledger：8 项 CPU 测试。
- 真实 Torch CPU Solver：6 项，通过 HiPerGator 既有 torch 2.8.0+cu128 环境执行。mixed binary group 真正移动参数；warm AdamW 的 all-zero/all-one group 保持参数、优化器状态和 scheduler 完全一致。
- Benchmark 与真实输出提取：15 项远端 CPU 测试，无 skip；原 vendor 参考程序验证 0–14 长度全枚举 32,767 条、至 64 长度额外 5,998 条、正式 suite 的 11,520 个测试，全部 exact pass。错 prefix 负对照失败，四色输入 Y 的反例说明新旧域确实不同。

合计 83 项 CPU 测试，另有 16 项只读真实模型/adapter/manifest preflight 检查，以及一个四分支合成集成轮。统一 runner 的实际执行、source SHA 和 GPU 原始记录复算见 `FINAL_VALIDATION.json`。

这些 CPU 数字是工程测试与参考程序执行，不能加入模型成功率表。集成轮的合成条件计数也单独标记，不替代真实训练。统一重跑命令是 `bash RUN_CPU_TESTS.sh`；它为每次运行新建日志目录，不截断旧日志。

## 实际修改与保留

本目录都是独立新增文件。原审计的 `outputs/verge_review_20260908/patches/minimal_fixes.patch` 保留三处旧代码明确修复；本轮并未把新研究定义强行打进正在运行的旧流程。benchmark 初次每类 4 个测试的草稿与日志保存在 `benchmark/generated_24_draft`，正式版每类 6 个，未覆盖草稿作为最终结果。

预算 journal 记录 generation attempt、原始 token IDs/文本/verifier 身份、update 与 SHA 链。发生非法 backend 输出时，失败生成仍记原始记录及已知 token 成本；未知成本记 null，不记零。EOS 包含在实际生成数内，EOS 后 batch padding 排除。

## 当前工程接线边界

第一轮真实模型 smoke `41369486` 已完成，1 B200 的 Slurm Elapsed 为 129 秒（0.035833 B200-h）。实际生成 32 条、16,384 tokens，4 个全零组均精确跳过。原始 parse 0/32、full pass 0/32；它暴露了两个新入口的接口遗漏，不能当作新 benchmark 的 regime 结论：

1. `evaluate_program` 没有原工程已有的 `extract_program`，直接把说明文字送给 DSL parser。已复用原函数且 AST 完全一致，CPU 重放原 32 条得到 parse 1/32、full pass 0/32。原输出和原 summary 未改；预修复源码已按 smoke 的 SHA 保存。
2. 新 HF 入口没有使用原 `verge_repair_protocol.py` 的格式前缀 `PROGRAM_PREFIX`，且 cap 缩为 512。单独的 `live_smoke_prefill.py` 恢复这个 target-independent 代码格式接口，prefix 属于 prompt，loss/生成预算仍只统计原生 suffix；采用原 2048 cap，只补测 1 组 8 条。修复不恢复旧 reward、teacher 路径或训练预算。

补测作业为 `41370126`，申请 1 B200、64G、17 分钟硬上限；加上第一轮的 129 秒，总硬上限 1149 秒，小于本轮已说明的 20 GPU 分钟。两次作业输出与成本分别记录，最终状态见 `runs/gpu_smoke*` 与 `FINAL_VALIDATION.json`。summary 内 `b200_hours` 是 Python 主区段计时，正式资源账使用 sacct 的 Elapsed。

两作业现均 `COMPLETED / ExitCode=0:0`。补测实际 113 秒，生成 8 条、12,483 native suffix tokens，parse 0/8、full pass 0/8，1 个全零组精确跳过；source adapter 未变。总计 242 单卡秒，即 0.067222 B200-h。两次未出现真实 backbone 的混奖更新，不能称已测到该模型的有效学习；只有 CPU TinyLM 测试实际覆盖了非零更新路径。补测原始错误涉及规范结束节点缺失、重复节点 ID、不存在的 `PULLER_AB` 等；不自动修写模型答案来取得通过。

新的数学 core、raw-evidence 接口、stage validator、buffer 与预算运行器已实现。CPU 集成使用显式合成 backend；真实模型 smoke 只覆盖初始 Solver，不伪称已经运行完整自动课程搜索或 Challenger RFT。

全模型正式闭环还需要把独立模块与 production generation、动态 stage dataset、guided proposal 和 Challenger RFT trainer 接成同一版本入口，并经一次小规模真实 round 验收。现有阶段数据生成/提示库是接口实现，不能把 Python 函数执行通过称为 Manufactoria 策略已能学会这些任务。这个剩余工程工作已经有明确定义，无须再等待不存在的旧稿。

正式闭环不再因类别表和 Delta 来源缺失而停工；它的执行顺序仍是确认候选 benchmark 的 regime、可点燃 stage、然后单轮与后续 gates。旧运行的零成功结果不自动成为新 benchmark 的反例或本版本的验证。
