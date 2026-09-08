# VERGE：全新 agent 执行交接

## 本轮任务

你是新接手 VERGE 的研究与工程 agent。用户怀疑旧 Codex 会话发生了上下文混淆，要求重新做最新任务。

从同目录的 `VERGE_own_method_design.md` 全文重新建立任务理解，独立复核该设计是否合理，核对真实仓库中的实现与执行状态，并实际完成可以明确完成的核查、最小修复和测试。不要停在复述方法或列待办，也不要把“重新接手”解释为删除旧成果或重训所有实验。

本交接新增的是执行边界和交付要求。它没有批准修改方法。`VERGE_own_method_design.md` 是当前待复核设计的原始依据；文件中的预测、存在性保证和有效性主张仍须核查，不能当成已完成的实验结论。

## 一、上下文与来源边界

1. 按以下用途区分材料：
   - 最新方法：本次提供的 `VERGE_own_method_design.md`。
   - 当前工程事实：现场读取的代码、配置、数据 manifest、checkpoint 元数据和原始日志。
   - 当前任务边界：本交接及用户在新会话中的直接指令。
   - 历史材料：只按最新文件明确引用或当前核查所需，定点读取；不得自动恢复旧会话的整套设定。
2. 遵守系统、安全、权限及适用的仓库工程约束。检查实际生效的 `AGENTS.md` / `AGENTS.override.md` 等指令文件；其中陈旧的科研设定若与当前方法冲突，记录冲突，不静默拼接，也不自行删除或改写全局指令。
3. 本任务是 VERGE。CAFD 的教师路径、相对 logit 目标、纯 KD 对照、T1/T2、MPC、KD/RL 路由不属于本任务。不要因共享目录、模型或环境代码而继承这些训练目标。UPLIFT、Logic-RAG 和 robotics 的设定也不进入本轮。
4. 不把旧 agent 的总结、自动记忆或上一轮 ChatGPT 的建议当作用户已确认的修改。没有原始依据的“已锁定”“已完成”“已经验证”均标为未核实。
5. 保留原文的 Challenger、Solver、target、stage、curriculum、frontier condition、archive 等术语。不要自行换方法名或统一到其他项目的角色名。
6. 不根据记忆指定模型家族、checkpoint、仓库位置或 SSH 地址。相同模型在别的项目中可用，不构成当前 VERGE 的模型选择依据。

## 二、先定位现场，保留旧成果

在用户已授权的当前工作区内，读取仓库根目录、当前分支和未提交改动；定位 VERGE 的实际入口、配置、测试、数据与日志。若当前目录并非目标工程，只沿已授权工作区及明确路径线索查找；不要搜索无关私人目录，不要猜测远端连接。

确认这些事实：

- 真实 target family、stratum、训练/selection/held-out/test 划分与 verifier 实现。
- 实际 backbone、revision、tokenizer、初始 checkpoint、两个角色的 adapter 和加载状态。
- 已有实现采用的是当前设计、旧版 VERGE，还是混合配置。
- 哪些运行真正完成；每项结果的模型、数据、奖励、预算口径与当前设计是否一致。
- 可用计算资源、当前已有作业，以及本次工作的授权范围。计算资源上限为两张 B200；使用资源仍须服从现场的实际配额和已获授权。

在仓库惯例允许的位置创建一个独立交接/运行目录，记录其实际路径。保留已有代码改动、数据、checkpoint、日志和正在运行的任务；不执行破坏性 Git 操作，不取消其他作业，不自行租用资源。没有依据时，不将旧结果补进新设计的结果表。

## 三、按原文重建当前方法

下面只是检索导航。完整原文始终保留在同目录；发现本节与原文不符时，记录差异并以原文为当前设计依据。

### 3.1 固定目标与条件（原文 §1.1）

逐测试 pass 为 0/1；超时、执行错误、routed to NONE、输出不符都按该测试失败。`f(x,y)` 是各测试类 exact-pass fraction 的最小值；`V_T = 1[f=1]`。

条件为 parse、`f >= theta`，以及 full pass；阈值原文为 `{1/|T_tests|, 0.25, 0.5, 0.75}`。frontier 为最低未饱和条件，`delta=0.1`。mean-LCP 仅记录，不进入 reward。

不要恢复旧稿的全测试 termination 前置门槛或 prefix/LCP 奖励。核查阈值、测试类分母及 parse/执行错误的处理是否与定义一致；空测试类、缺失输出或非法阈值不得静默忽略。

### 3.2 三类 stage 与 validator（原文 §1.2）

三类来源：注册族参数；Challenger 提出的受限 tape 变换或识别谓词；带 hint 的 target。

hint 的来源是 archive 已验证程序库中相关 stage 实例的解，或较小实例的解。target 的最终评估不附 hint。

validator 的四项为：可执行、非常量、无泄漏、hint-regret。原文的 hint 探针对比条件是 `p(with hint) > p(without hint)` 且 `p(without hint) < 1-delta`，规模为 32 题 × 8 条、两次。若某项检查对非 hint stage 的适用方式不明确，先标出歧义并核对实现，不自行改变四项的定义。

Challenger 提出的可执行规格属于不可信输入；不得以 agent 自身的系统权限任意运行。优先检查工程已有的受限表达、执行隔离与资源限制。这个要求是执行安全边界，不改变四项科研 validator 的定义。

### 3.3 Checkpoint archive（原文 §1.3）

key 为 `(b,s)`。`s=1` 表示 post-training batch 中 parse-valid 程序的节点图含环比例达到 0.5；否则为 0。每格最多一个 incumbent。

空格直接写入；非空格要求候选在对应 frontier rate 上显著优于 incumbent，比较使用 selection instances 上的 paired bootstrap。原文 lead 是最高 rung 的 incumbent，同 rung 取 frontier rate 较高者。

保留“多个格子提供后续训练起点”的设计。不能未经批准退回单一血统，也不能静默改写 lead 的定义。parse-valid 数量为零时的结构描述子、并列规则和 incumbent 评估缓存等实现细节若没有依据，列为待核实项。

### 3.4 配对 Solver 分支（原文 §1.4）

每轮共用一个由 Challenger 选择的 base。G 条 curriculum branch 与一条 direct branch 均由它复制，使用 fresh optimizer、原文 `beta=0`、weight decay 为零。核对代码中 beta 的具体含义，不能混用其他版本的同名参数。

预算 B 的原文口径是 generated tokens。L 个 stage 各占 `(1-eta)B/L`；stage 内分布为 `alpha D_T + (1-alpha)D_u`，尾段 `eta B` 训练 target。direct branch 使用相同预算直接训练 target。

Solver reward 仅为当前 prompt 所属任务的 binary full pass。全组 reward 相同的 update 精确跳过。每个 stage 后进行无 hint target 探针，每题 8 条；最终 post-training batch 每题 32 条，用于 condition rates、success counts 和结构描述子。

不要将 partial verification 加进 Solver loss；不要把旧版 training-token 预算自动替代当前 generated-token 预算；不要为不同 curriculum 另选不同 base 却共享一个 direct 对照。

### 3.5 课程评价、Challenger 学习与一轮流程（原文 §1.5–§2）

至少两个 branch 的 post-training target successes 各达到 q 时，`kappa_t=m`；否则 `kappa_t=b(base)`。

`Gamma_g = rho_hat_kappa(S_g) - rho_hat_kappa(S_0)`；J+ 要求 gain 为正且 paired CI 不包含零。这里 `S_0` 指本轮 direct branch，不是 CAFD 的固定初始学生。

Challenger 使用同 backbone 的独立 LoRA，读取 target、全部 incumbents 及 archive，输出一个共享 base 和 G 条有序 curricula。学习由 archive in-context 与累计 J+ 样本上的 rejection-sampling fine-tuning 组成；达到 N_min 条新增样本时执行一遍。当前设计不使用在线 Challenger GRPO。

依次完成 proposal/validation、配对训练、评价、archive 更新、经验追加及满足条件的 proposer fine-tuning、事件记录。分别记录 first success、reward switch、learning onset、格子占用与非 lead base 事件。原文终止条件为 T 轮或 lead full-pass rate 达到 `1-delta`。

## 四、独立复核：不预设通过，也不擅自修订

先独立形成判断，再提出最小修订。每个实质问题注明：原文位置、数学或代码依据、实际影响、可执行检验、建议，以及是否需要用户决定。

优先检查下列接口。这些是待检验的问题，不是已批准的改法：

1. 最弱测试类的 fraction 在真实分支上是否产生可区分信号？“first success 前非退化”是否有实际证据？必要性本身不能被当成有用性或非退化性的证明。
2. 分格 archive 保留低分候选与 J+ 正例 fine-tuning 是否一致？系统能否学习最终有价值但即时 gain 为负的前序课程？重复的局部改善是否真正改变 archive？
3. hint-regret 接受的 stage 是否有含不同 reward 的组、有效 Solver 更新，以及无 hint 目标迁移？成功率提高、可发生更新、迁移有效应分别核查。
4. 原文 lead 规则在部分指标与 full-pass 表现不一致时会选哪个 checkpoint？节点图含环是否仅是静态结构描述？每格单 incumbent 实际能保留哪些回退路径？
5. paired bootstrap 的抽样单位、训练 seed 的独立性、跨 seed proposer 经验共享，以及 P1–P7 的判据是否支持各自的主张？
6. “per-stage Delta 与 telescoping 恒等式同修复版”依赖什么具体文件？只定位该定义及配套实现，不把被引用旧稿整体升级为方法依据。找不到时明确标注缺失，不能自造公式填空。
7. first success、reward switch、learning onset 分别表示什么？本文件没有提供 grokking 的完整判据；未核验训练/泛化动态时，不将这些事件直接命名为 grokking。

上述问题不自动触发新增 hindsight credit、成功路径回放、另一种 lead selection、另一组 validator 或新模型。涉及研究设计变化时，单独提出；本轮不悄悄合并。

## 五、实际执行边界

### 本轮直接完成

完整阅读、方法复核、仓库与配置对照、已有结果核验、CPU 单元测试，以及不改变科研定义的明确实现修复。修复应最小化、可审阅、带测试。即使某个定义或 checkpoint 缺失，也继续完成不依赖它的工作。

有已授权 GPU 资源、真实配置且不依赖待定方法选择时，可以运行有明确上限的最小 smoke test。事前说明实际范围与估计成本；它只验证加载、生成、verifier、更新与日志链路，不能冒充正式实验。缺少预算授权时，完成 CPU/mock 测试和可运行配置，不提交 GPU 作业。

### 本轮不自动扩张

不因“全新 agent”而删除历史、重训全部模型、重复已有高成本运行或重新筛选目标；不将全文中的 Gate 0–3 当作本轮获准全部提交的作业。保留原文执行顺序，正式训练只在参数、数据、预算和授权明确后继续。

原文未写死的 backbone、checkpoint、target stratum、B、alpha、eta、q、N_min、T、置信区间细节等，先查真实工程中的明确依据。仍不明确时，区分“阻塞正式运行”与“不阻塞本轮核查”，一次性提出最少的必要问题。不得从 CAFD 或旧稿自动填值。

不要为得到正结果改分母、丢弃失败样本、替换目标、调整阈值或将开发集结果包装为独立测试结果。保存失败记录、原始输出与实际成本。

## 六、最小验收与交付

至少检验：

- 单测试 timeout 不抹掉其他测试记录；各测试类分母固定；f、V_T 与条件序列按原文计算。
- mean-LCP 不进入 reward；Solver 的 branch reward 全为 0/1；常数 reward update 跳过后参数保持不变。涉及 optimizer state 的行为要实测，不只读配置。
- G+1 个 branch 共用一个 base、预算口径一致；hint 仅在相应训练 prompt 中出现；target 评估没有 hint。
- reward switch 依赖至少两个 branch；J+ 的 CI 条件正确；结构格子、替换与 lead 按原文执行。
- Challenger 不走在线 GRPO；经验样本来源可追踪；尚未达到 N_min 时不触发 fine-tuning。

在独立目录交付三份简洁文档，避免另造一套庞大项目规范：

1. `METHOD_REVIEW.md`：先给独立结论，随后给依据、问题与最小建议；原文决定、实现事实、未批准建议、待测预测分开。
2. `IMPLEMENTATION_AUDIT.md`：每个方法组件对应的真实路径、当前状态、差异、测试命令与结果；附最小代码改动或 diff 路径。
3. `NEXT_ACTION.md`：已完成什么、没有运行什么、剩余必需决定、下一条可复制命令及资源需求。命令没有实际验证时明确注明。

代码、配置与日志遵循仓库已有组织，不覆盖旧结果。没有真实仓库访问时，仍完成基于原文的方法复核，并明确代码层面尚不可验证；不要声称已完成实现或训练。

第一条回复简短说明当前项目、原始依据、可访问工作区和本轮执行范围，然后立即开始读取与核查。最终回复给实际产物、测试结果及最少的阻塞问题，不以泛泛的“后续可以做什么”结束。
