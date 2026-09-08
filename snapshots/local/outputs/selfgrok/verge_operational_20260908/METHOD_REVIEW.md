# VERGE：可执行版本复核，2026-09-08

现在按用户最新授权推进：允许更换 benchmark、补齐操作性定义，研究问题与 scope 保持。两项此前缺少来源的定义不再使工程停工；本目录实现的是新版本，原稿、旧工程和旧实验继续保留。实验应当能执行到明确结论，P1–P7 是否成立由数据决定。

## 保持的研究问题与范围

固定 target 的 binary full-pass 稀疏奖励下，Challenger 能否通过 target verifier 的必要条件、课程任务选择和分格 checkpoint 选择，帮助 Solver 获得无 hint 的目标能力。保持：

- Solver 只接收其当前任务的 binary full pass；beta=0、wd=0，同奖励组精确跳过。
- Challenger 与 Solver 同 backbone、独立 LoRA；每轮同一个 base，3 条课程加 1 条 matched-generated-token direct。
- 三类 stage 全部保留：注册族、受限 tape 变换、带 hint 的 target；四项 validator 全部保留。
- min-per-class exact fraction、原来的四个阈值、六条件、reward switch、Gamma/J+ 和跨 seed 正例 RFT。
- `(frontier rung, static cycle descriptor)` 分格，允许选择非 lead base；无 hint 的 stage probe 和最终评估。
- P1–P7、direct/dense、SOAR-reward、uncertainty、random/frozen/irrelevant、order/pooled 等原文研究比较不删除。第一条成功不直接命名为 grokking。
- 不增加 CAFD 教师路径、T1/T2、KD/RL 路由、MPC 或 Solver partial reward。

## Benchmark 调整：保留变换，改变输入域

原实际目标为 `RBYGRB + input.replace("RB", "YG")`。新候选 `verge_prepend_rb_v1_20260908` 保留这个输出函数、Manufactoria DSL 和四符号执行环境，把输入域固定为 `{R,B}`、长度 0–64。`Y/G` 可作工作符号及输出符号。原来的四符号输入 benchmark 保留在旧目录，结果不合并。

这使我们能构造一个 18 节点循环程序，证明新任务存在小型可执行解。该程序只作 CPU verifier 验证，不进入 Solver 模仿数据，也不自动成为 hint。是否存在可点燃 stage、无 hint 转移、是否 direct 已足够，仍由原 Gate 0/1 测量。输入域调整不能写成原四符号任务已解决。

新增类别由变换的语义确定：非重叠 `RB` 匹配次数为 `0/1/2plus`，分别与末尾是否为待处理 `R` 交叉，共六类。空串属于 `matches_0__pending_r_0`。每 suite 每类 6 个不同输入，共 36 个测试；每个分母固定为 6，空类、缺类或错分组拒绝。

选择 6 个而非初稿 4 个有具体原因：每类 4 个时，`f>=1/24` 与 `f>=0.25` 恒等，损失一个 rung。每类 6 个时，原阈值 `1/36,0.25,0.5,0.75` 对应每类至少 `1,2,3,5` 个通过，事件各不相同。没有改变阈值或合并类别。

四个 split 的输入所属由版本号和输入 tape 的 SHA256 预先固定，训练/selection/heldout/test 的输入交集为零；分别有 128/64/64/64 个 suite。相同 split 的 suite 可以复用输入。目标函数参数与提示相同，因此这不构成独立任务族的泛化测试。未读取 heldout/test 模型结果来选类或改 benchmark。

## 本次新补齐的操作性定义

这些是用户放宽后采用的新定义，不能称为恢复出的旧稿。

| 项目 | v1 约定 | 保持的原文内容 |
|---|---|---|
| 完整类别定义 | 上述六类、每类 6 个；显式类别全集 | 最弱类 fraction、固定分母 |
| per-stage Delta | 固定本轮 kappa，在相同生成 token 边界计算课程与 direct 的 rate gap，再对 gap 作相邻差 | 完整课程 Gamma 是最终课程减 direct |
| tail 与样本口径 | 分解包含 target-only tail；8 样本 probe 的差望远镜相消；另记 `Gamma_32-gap_final_8` 采样修正 | probe 每题 8 条、final 每题 32 条 |
| direct probes | 在所有课程阶段边界的并集保存 direct checkpoint/probe；最终前 8 条直接引用 final 的原始记录 | 同 base、相同 token 里程碑、同 selection |
| 饱和 frontier | `b=7`；比较和报告使用 full-pass rate | 六条件与 lead 优先更高 rung |
| 无 parse 的结构格 | `s=0`，同时显式保留 parse_valid_count=0 | 静态图环比例；不将 0 称为已证无环 |
| archive ties | 真正同分保持先入者；paired bootstrap 按 selection instance、95% percentile、2000 次 | 同格显著改善才替换 |
| hint-regret | with/without 各 32×8，共 512；两臂共享题目和 seed 索引 | 两个成功率条件不增加第五 gate |
| 非 hint 的第四项 | 明确记为 NA，其余三项仍须通过 | 四项 validator 接口不增删 |
| generated B 边界 | 先限制生成 cap；剩余不足 8 token 时按同阶段分布生成并记 drain，不学习；每阶段最多 7 token | 完整组、完整程序、实际生成预算相等 |
| N_min | 开发配置为 8 个新增 J+ 正例；一次训练累计正例一遍 | 未达阈值不更新；失败不吞计数 |

hint 探针原文“32×8、两次”与每新 stage 512 rollouts 的预算有歧义。这里将“两次”明确操作化为 with/without 两臂。若另做两组重复，需要另记 1024 的真实成本。

防泄漏比较的实例键由冻结 manifest 的 target 定义与 suite 输入集合导出，不能靠不同 split 名、随机 id 或 Challenger 自报元组获得通过。同一函数定义跨 target splits 重用是 benchmark 的事实；suite 与输入级别的边界必须另外保留。

## 方法合理性判断

这个版本具有可执行接口和可检查的失败状态，值得先做有限的 regime/可点燃性试验。循环程序的存在解决了任务可解性；它没有证明当前 backbone 能从 binary reward 点燃，也没有证明自动搜索优于 direct。min-per-class 可以区分某些 partial solutions，但仍可能让所有候选得零，不能继续把“first success 前非退化”写成无条件定理。

本轮真实 smoke 已给出一个具体开发方向：修复代码块提取和恢复原代码格式前缀后，8 条 cap 2048 的补测仍全部 parse 失败，包含缺少规范 `END end`、重名节点和不存在的节点类型。它们尚未形成可执行程序，所以不能据此判定新变换太难或课程无效。接下来按原 Gate 0/1 先测简单注册 stage 的语法有效率、binary 成功与 mixed-group 更新，再测向无 hint target 的转移。不要直接花完整闭环预算在未验证的点燃条件上。

archive 可以保存低 rung 的循环分支，J+ RFT 却只学习本轮显著正增益；保留这两个机制，实际记录绕路与长期 lineage。静态环只作 descriptor，不直接等同算法性的循环使用。原 lead 规则也保留，通过 full-pass 曲线同时暴露其可能选到低 full-pass checkpoint 的情况。

P1 应要求正向预测的区间在 0.5 上方；只排除 0.5 可能得到反向预测。P2 的单条 lineage 是机制证据，不代替与单血统对照。P3 的成功率改善、含混合 reward 的组、真正更新、无 hint 转移分别记录。P5 的跨 seed 经验共享必须在训练组与评估组的划分中处理，不能把共享同一 Challenger 的种子当完全独立实验。这些检验保留原问题，不通过删掉失败样本来获得结论。

## 来源与工程指令冲突

原文依据是两个用户提供文件；上一轮审计保留了它们的逐字副本与 SHA256。现场没有发现适用的 AGENTS.md/override。旧工程的 11 条件、KL=.02、loss-token B、单血统和 Challenger 在线策略梯度属于历史实现，不与本目录拼接成新定义。

原交接文件说“没有批准修改方法”，最新用户直接授权放宽并允许更换 benchmark；本轮以上操作性补充依照最新指令。文件资源上限为两卡，用户直接指定为 4 B200、192G；本轮新增 smoke 只申请 1 B200、64G、20 分钟。正式 Gate 0–3 的全套费用不由此自动提交。
