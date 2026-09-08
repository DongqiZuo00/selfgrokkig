# VERGE 可执行协议核心（2026-09-08）

这是独立 CPU 协议库，不是已经完成接线的 GPU 训练器。原始依据为本轮完整读取的 `VERGE_own_method_design.md`，方法命名、研究问题与 scope 保留。用户最新授权允许补齐方法和调整 benchmark；以下新增约定使用该授权，不称为已恢复的旧稿定义。测试数据均为合成 fixture，不是正式测试分组、真实模型结果或有效性证据。

## 原文规则的实现

- `TestClasses(test_ids, class_by_test, expected_classes=...).evaluate(parsed, outcomes)` 要求显式完整的 test → class 分区，使用每类固定分母以及类通过率的最小值；缺失输出、timeout、error、NONE 均由调用方规范成该测试的 `None` 或 `False`，仍保留分母。少了整个测试结果槽则报错。正式 benchmark 必须传入冻结的全部 `expected_classes`，防止整个类别消失后仍通过映射检查；这个参数可省略仅供已有调用兼容，省略时库无法发现外部预期却未声明的类。返回六项条件：parse、`f>=1/N`、`f>=0.25`、`f>=0.5`、`f>=0.75`、full pass。`binary_reward` 只返回 full pass 的 0/1。
- `StageSpec(kind, spec)` 保留 `registered`、`tape_transform`、`hinted_target` 三类接口。它只是类型化规格，不执行不可信字符串，也不声称已经完成四项 validator。
- `plan_round(base_checkpoint, curricula, B, eta)` 只接受一个共享 base，创建 G 条课程与一条 direct 的计划；fresh optimizer、beta=0、weight decay=0、binary Solver reward 是显式计划字段。B 的单位为 **generated tokens**，每条课程包含 target-only tail。实际优化器创建/常数 reward 跳过、alpha 混合采样及 tokenizer 计费须由训练器接线，计划声明本身不是训练验证。
- `Evaluation(checkpoint, training_seed, manifest_id, instance_ids, sample_ids, counts)` 保存同一训练 seed 上固定 selection instances 的六项整数条件通过计数。`counts[i][j]` 是第 i 个实例上满足第 j+1 个条件的 rollout 数。Final 每题 32 条，stage probe 每题 8 条；rate 先除以固定 rollout 数，再在 selection instances 上取均值。
- `score_round(base, direct, curricula, q)` 从所有 G+1 条 final batch 选择本轮共享 kappa，然后计算每条课程相对 direct 的完整 Gamma、paired CI、J+。`q` 是各 branch 的 full-pass rollout 总数阈值。至少两条达到 q 才 switch；否则使用 base 的 frontier。此处“两条 branch”包含 direct，且 switch 是每轮重新计算，不因历史 switch 永久锁定。
- `paired_gain` 以 **selection instance** 为 paired bootstrap 抽样单位，在单个训练 seed 内计算不确定性；训练 seed 和每题 rollout 都不冒充独立实例。跨训练 seed 的结论需要单独统计。95% percentile CI、2,000 次重采样和固定 PRNG seed=0 是本次明确默认值，可在运行前配置冻结；`J+` 要求 estimate>0 且 CI 下界>0。
- `CheckpointArchive.consider(Candidate(evaluation, cycle_count))` 按 `(b,s)` 写入，空格直接写，同格显著改善才能替换；不同格可保留回退路径。`lead` 始终按最高 b，再按该 frontier rate 排序。Direct 的候选同样可调用 `consider`；是否“移动”须由训练器的参数/检查点事实决定，库不假称已训练。

## 本次补齐的定义

1. **条件序列最小测试数**：本协议要求每个目标测试集至少 4 个测试，保证原文 `1/N <= 0.25`，保留原文六项位置和阈值。N=4 时前两个 f 阈值相同，允许重复必要条件。所有 `expected_classes` 必须有实际成员，空类、缺失类或未声明的额外类均拒绝。
2. **整数预算**：tail=`floor(eta*B)`；剩余 token 在 L 个 stage 间均分，余数逐个分给较早 stage。每段均需正预算，计划保证每条 branch 的总训练预算正好 B。实际训练不能用旧 `train_tokens` 或 padding 数代替 generated tokens。
3. **终态与结构空分母**：所有条件饱和时记 `b=m+1=7`；该终态格比较 `rho_m`。这不改变原文 lead 先比 rung 的规则。零 parse-valid 时用 `s=0`，同时保留 `parse_valid_count=0`，不把它解释为已证明程序无环。完全相同的 lead 排序值保留最先进入字典的格，替换相同值仍须满足显著改善。
4. **paired bootstrap 细节**：禁止混合训练 seed、manifest、selection IDs 或 rollout sample IDs 的配对；至少两个 selection instances 才估计实例间 CI。此 CI 对已经训练出的 checkpoint 条件化，不是对整个方法跨 seed 有效性的证明。
5. **per-stage Delta 与 telescoping**：采用下式，包含 target tail，固定同一 kappa 和同一评估样本。它是本次新增的记账分解；不是恢复被引用但缺失的“修复版”定义，也不是各 stage 的因果贡献。

对课程 g，令训练 generated-token 累计边界为 `t_0=0 < ... < t_L < t_(L+1)=B`。课程与 direct 在同一 token 边界保存 checkpoint；direct 需要所有课程边界的 **并集**。每个边界在相同 selection instances 和相同 8 个生成 sample IDs 的 P8 上评价，固定本轮最终选择的 kappa：

```text
d_l = rho_kappa(course checkpoint at t_l; P8)
      - rho_kappa(direct checkpoint at t_l; P8)
Delta_l = d_l - d_(l-1), l=1,...,L+1
d_0 = 0   (same base checkpoint and identical cached base evaluation)
Gamma = rho_kappa(course final; P32) - rho_kappa(direct final; P32)
sampling_correction = Gamma - d_(L+1)
Gamma = sum_l Delta_l + sampling_correction
```

最后一项 Delta 对应 target-only tail。P8 必须为最终 P32 的前 8 个冻结 sample IDs。Final checkpoint 的 P8 直接从 P32 的原始 rollout 记录取前 8 条，不能另行采样后仅改 ID。`decompose_stages` 核对样本 ID、checkpoint、共同起点、计数子集可行性和恒等式；完整前缀来源仍需训练日志证明。Challenger context 应保留全部 Delta、`sampling_correction`、kappa、边界、sample/manifest 身份和最终 Gamma，不能把 correction 隐去，也不能把缺失值置零。

新增评估成本单独记录：direct 在 `union(milestones) - {0,B}` 的每个内部边界需要 `8 * N_selection` 条 target rollouts；base 若没有可复用 P8 缓存，还需 `8 * N_selection` 条。Final P8 来自已经计划的 32 条，不增加 rollout。课程原有每 stage 的 8 条不变。所有探针的实际 generated tokens 另计入评估总成本、GPU 时间和资源预算，**不**挤入每条 branch 的训练 B，也不能声称这些探针完全免费。

`round_evidence.py` 提供原始记录到计数的实际接口：

```python
from round_evidence import RawRollout, aggregate_rollouts

# 每条实际输出一个记录；conditions来自冻结verifier，has_cycle是解析后的静态图事实。
# RawRollout(checkpoint, training_seed, manifest_id, instance_id, sample_id,
#            conditions=(bool,)*6, has_cycle=bool_or_None)
final = aggregate_rollouts(raw_records, checkpoint=checkpoint,
                          training_seed=training_seed, manifest_id=selection_manifest_id,
                          instance_ids=frozen_instance_ids, sample_ids=frozen_32_sample_ids)
probe_after_tail = final.first_eight()  # 对相同RawRollout对象取前8个slot；没有重新生成。
evaluation = final.evaluation
candidate = final.candidate
```

聚合器要求每个冻结 instance × sample 恰好一条记录，检查 checkpoint、training seed、manifest、六个嵌套 bool 条件，拒绝缺失/重复/额外样本。Parse-valid 的 `has_cycle` 必须显式 bool，parse-invalid 必须 None。库据原始记录统计静态环数；它不会把静态含环描述解释为该环在执行时被使用。输入可为 `RawRollout` 或相同字段的字典；输出按冻结 ID 顺序排列。每个 final 的 `first_eight()` 来自相同原始记录对象，消除“另采一批却自称前缀”的接口误用。条件 bits 本身仍需通过真实 verifier 的运行证据追溯；聚合器不能替代 verifier。

## 执行与边界

本地执行过的命令（Windows PowerShell，工作目录为本次 `hipergator-x20`）：

```powershell
& 'C:\Users\ddong\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest discover -s 'outputs/selfgrok/verge_operational_20260908/protocol' -p 'test_*.py' -v
```

结果：31 个 CPU 合成契约测试通过（协议 25、原始证据聚合 6）；完整输出见 `cpu_protocol_tests.log`。无 GPU 作业，无真实模型训练结果。未修改原始工程、旧 checkpoint、日志或作业。

从任意 CPU Python 环境运行可复制命令：进入本目录后执行 `python -B -m unittest discover -p 'test_*.py' -v`。除 Python 标准库外无依赖。

未在此小库中实现训练框架、程序执行器、真实测试分组生成器、hint-regret validator、Challenger LoRA/SFT 或 GPU runner；这些必须在真实工程中逐项接线并验证。此文件的配置/计数测试不能替代优化器状态不变测试、真实 verifier replay 或一次获授权的有界训练 smoke test。
