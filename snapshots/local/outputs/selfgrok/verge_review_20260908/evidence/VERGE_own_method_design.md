# 我们自己的方法：设计、理由、可证伪的预测

术语沿用稿子：target $T$、verifier $V_T$、conditions $c_{1:m}$、frontier condition $b$、Challenger $Q_\theta$（下文按其新职能也叫 proposer，进稿时统一为 Challenger）、Solver $S_\phi$、curriculum $\mathcal{C}$、stage $u$、budget $B$、direct branch $S_0$、reward $\Gamma$、reward condition $\kappa$、$\mathcal{J}^+$、archive $\mathcal{M}$。新名字只是占位。

---

## 0. 六条设计决定，每条由一个约束逼出

| 约束（前面各轮确立的事实） | 设计决定 |
|---|---|
| first success 之前 outcome 的任何函数在所有 branch 上同值（Prop. A）；唯一的目标信息是 verifier 内部的逐测试结果 | proposer 的 reward 只在**冻结的 target verifier** 上量：per-class exact-pass fraction $f$ 的 frontier gain，compute-matched |
| mean-LCP 级序偏向退化程序；累积乘积门槛惩罚 loop | 条件序列只有 parse、$f\ge\theta$、full pass；超时按测试计失败 |
| 用代理量选 checkpoint 就是在塑造 Solver 轨迹，且必要条件的顺序 ≠ 学习路径的顺序（unrolled lookahead 是 $f$ 的局部最优，学 loop 要先掉下去） | 明写方法是**选择式优化器**；把单一血统改成按 (frontier rung, 程序结构) 分格的 checkpoint archive，允许绕路 |
| 每个 reward 样本是一次完整 Solver 训练，两卡几天只有几十到一百个样本 | proposer 不做在线 GRPO；学习 = archive in-context + 跨 seed 的 rejection-sampling fine-tuning |
| 注册族是 family × tier 的格子，不保证含有缺失子技能；reward 在 target 上量之后 stage 的 verifier 只需可执行、不需可信 | stage 空间三类：注册族参数、proposer 写的 tape 变换、带 hint 的 target；validator 只做可执行 / 非常量 / 泄漏 / hint-regret 四项检查 |
| 目标能否点燃不由 reward 设计决定 | 闭环前置 gate；claim 全部写成可证伪的预测 |

---

## 1. 组件

### 1.1 目标与条件序列

逐测试 $\mathrm{pass}(\tau)\in\{0,1\}$（超时、执行错误、routed to NONE、输出不符均为 0）。$f(x,y)=\min_k$ 各测试类通过率；$V_T=\mathbb{1}[f=1]$。条件：$a_1$ = parse，$a_j=\mathbb{1}[f\ge\theta_j]$，$\Theta=\{1/|\mathcal{T}|,0.25,0.5,0.75\}$，$c_m=V_T$。嵌套由 $\theta$ 单调与"$f>0$ 蕴含 parse"自动成立。$\rho_j(S)$、$b(S)=\min\{j:\rho_j<1-\delta\}$、$\delta=0.1$。mean-LCP 只记录，不进任何 reward。

### 1.2 Stage 空间与 validator

一个 stage 是可执行任务分布加其 binary verifier，三类来源：

- **注册族**：$(\text{family},\ \text{tier},\ \text{alphabet},\ \text{length range},\ \text{mutation tier})$，测试协议按族冻结。
- **proposer 写的 tape 变换**：期望输出函数 $g:\Sigma^*\to\Sigma^*$（或识别谓词），限制在字符串操作子集内；输入分布（字母表、长度范围、corner cases）；测试由冻结协议从 $g$ 生成。
- **带 hint 的 target**：target 实例的 prompt 附一个 hint——从 archive 的已验证程序库里取的相关 stage 实例的解、或一个更小实例的解；verifier 是 target 自己的。这是唯一能**保证存在可学 stage** 的来源：只要某个 hint 让当前 Solver 在 hinted target 上的成功率离开零，就有一个与 target 相邻的可学任务；hint 强度递减就是一条 curriculum 轴。

validator 对每个 stage 做四项检查，不多不少：可执行；非常量（corner cases 上至少两种期望输出 / 两种判定）；无泄漏（实例参数元组不与 selection / held-out / test 重合）；hint-regret 探针 $\hat p(\text{with hint})>\hat p(\text{without})$ 且 $\hat p(\text{without})<1-\delta$（32 题 × 8 条，两次）。SPADE 式的 regret 只在这里出现，不是 reward。

### 1.3 Checkpoint archive（分格）

格子 key $=(b,\ s)$：$b$ 为 frontier rung，$s\in\{0,1\}$ 为程序结构描述子——post-training batch 中 parse-valid 程序里节点图含环的比例 $\ge0.5$ 记 1。每格最多一个 incumbent。写入规则：branch $S_g$ 落入 key$(S_g)$；空格直接写入；非空格则 $\widehat\rho_b(S_g)$ 显著高于 incumbent（selection instances 上 paired bootstrap）才替换。lead $=$ 最高 rung 的 incumbent，并列取 $\widehat\rho_b$ 高者；lead 是报告用的"当前 Solver"，也是终止时的输出。

为什么要分格：$f$ 是必要条件不是学习路径。unrolled lookahead 把 $f$ 推到 7/18 后是局部最优，带 marker 的 loop 在学会之前 $f$ 更低。单一血统按 $f$ 贪心会把 loop 分支淘汰；分格让它在 $(b, s{=}1)$ 里活着，proposer 可以从它继续。每轮分支数不变，分格不增加算力。

### 1.4 Solver branch

一轮取一个 base（proposer 指定的某格 incumbent）；$G$ 条 curriculum branch 与 1 条 direct branch 都从 base 复制，fresh optimizer，$\beta=0$，wd 0。stage 按序训练，各占 $(1-\eta)B/L$ generated tokens，训练分布 $\alpha\mathcal{D}_T+(1-\alpha)\mathcal{D}_{u_\ell}$，reward 只有该 prompt 所属任务的 binary full pass（hinted target 的 full pass 也是 $V_T$，只是 prompt 带 hint）。全组同 reward 的 update 精确跳过。每个 stage 后做 target 探针（无 hint，每题 8 条）得 $\widehat\rho^{(\ell)}$；尾段 $\eta B$ 在 $\mathcal{D}_T$；最终 post-training batch 每题 32 条，得 $\widehat\rho(S_g)$、$n^{\mathrm{succ}}_g$、结构描述子 $s(S_g)$。direct branch 在 first success 之前由跳过规则精确等于 base。

### 1.5 Reward 与 $\mathcal{J}^+$

$\kappa_t=m$ 若至少两个 branch 有 $n^{\mathrm{succ}}\ge q$，否则 $\kappa_t=b(\text{base})$。$\widehat\Gamma_g=\widehat\rho_{\kappa_t}(S_g)-\widehat\rho_{\kappa_t}(S_0)$；per-stage $\Delta_{g,\ell}$ 与 telescoping 恒等式同修复版。$\mathcal{J}^+_t=\{g:\widehat\Gamma_g>0,\ \text{paired CI 不含 }0\}$。不再需要 lexicographic admissibility——回退的分支落到自己的格子里，不会覆盖高格的 incumbent。

### 1.6 Proposer

同 backbone 的独立 LoRA。context：target 描述与条件列表；全部 incumbent 的 $(b,s,\widehat\rho_{1:m})$；archive $\mathcal{M}$（base、curriculum、per-stage $\Delta$、$\widehat\Gamma$、$\mathcal{J}^+$ 标记、first-success 标记、已验证程序库的索引）。输出 JSON：base 格 + 有序 stage 列表（每个 stage 标类型与规格），guided decoding。学习两道：$\mathcal{M}$ 每轮追加；$\mathcal{B}_Q\leftarrow\{(\text{context}_t,\text{base},\mathcal{C}_g):g\in\mathcal{J}^+_t\}$ 跨 seed 汇总，新增 $\ge N_{\min}$ 条时做一遍 rejection-sampling fine-tuning。$\mathcal{J}^+_t=\emptyset$ 的轮不更新，计数。

### 1.7 事件

first success（采样事件：branch、rollout 序号、$\hat p_T$ 与区间）；reward switch（$\kappa_t=m$ 首轮）；learning onset（lead 的 $\hat p_T\ge0.05$ 首轮）；每格首次被占据的轮次；绕路事件（某轮 base 不是 lead）。

---

## 2. 一轮

1. proposer 读 incumbents 与 $\mathcal{M}_t$，输出 base 与 $\mathcal{C}_{1:G}$；validator 四项检查，hint-regret 探针。
2. 从 base 复制 $G+1$ 个 branch；训练（跳过规则）；每 stage 后 target 探针；最终 batch。
3. $\kappa_t$；$\widehat\Gamma_g$、$\Delta_{g,\ell}$；paired bootstrap；$\mathcal{J}^+_t$。
4. 每个 branch 按 key 写入 archive（空格直接写，非空格显著更高才替换）；direct branch 若移动了也参与；更新 lead。
5. $\mathcal{M}_{t+1}$ 追加；$\mathcal{B}_Q$ 追加 $\mathcal{J}^+$ 成员；达到 $N_{\min}$ 则 fine-tune proposer。
6. 记录事件。终止：$T$ 轮或 lead 的 $\widehat\rho_m\ge1-\delta$。输出 lead、全部 incumbents、$\mathcal{M}_T$、$Q_\theta$。

---

## 3. 这是谁的方法：与相邻工作的差别（只列已同行评审的）

| 相邻工作 | 它有的 | 我们不同的 |
|---|---|---|
| SOAR（ICML 2026） | target-anchored 的 teacher，outcome reward，单一血统 | reward 是 verifier 必要条件上的 compute-matched gain，在 outcome 为零时非退化；分格 archive |
| R-Zero（ICLR 2026）、Absolute Zero（NeurIPS 2025） | 自生成任务，target-blind | 环境设计被固定 target 的 verifier 锚定 |
| Unsupervised environment design（Dennis 2020, NeurIPS）、POET 一系（GECCO 2019） | regret 驱动、种群与 niche、环境-agent 协同进化，target-blind | regret 只做 validator；niche 由 target verifier 的 rung 与程序结构定义；fitness 是 target 上的必要条件 |
| RL Grokking Recipe（ICLR 2026） | Solver 端 dense per-test reward，人写 curriculum | per-test 信号在 proposer 端与 selection 端，Solver 只见 binary full pass；curriculum 由搜索得到 |

一句话：**target-anchored 的开放式 curriculum 搜索**——环境（stage）由 proposer 生成，fitness 是固定目标 verifier 的最低未饱和必要条件，搜索在按 verifier rung 与程序结构分格的 checkpoint 种群上进行，Solver 自始至终只在可执行任务的 binary full pass 上训练。

---

## 4. 可证伪的预测（每条给出 refuter；被否证的部件删掉，方法退化为对应的更简版本）

| 预测 | refuter | 否证后 |
|---|---|---|
| P1 frontier gain 预测 first success：每个 branch 的 $\widehat\Gamma$ 对 follow-on probe 下 time-to-first-success 的 concordance index 区间不含 0.5（按轮分层） | 区间含 0.5，或 target-irrelevant ladder 的 C-index 一样高 | ladder 不是搜索信号，方法只剩 best-of-$K$ 选择 |
| P2 分格有用：至少一条到达 first success 的血统经过 $s{=}1$ 且当时 $\widehat\rho_b$ 低于 lead 的格 | 所有到达 first success 的血统单调、从不换格 | 删分格，退回单血统 |
| P3 hinted target 可迁移：hinted stage 进入 $\mathcal{J}^+$ 的比例高于注册 stage，且无 hint 的 $f$-rung 上升 | hinted stage 从不进入 $\mathcal{J}^+$ | 删 hint 来源；存在性保证失效，方法退回注册族 + 自写变换 |
| P4 自写变换有用：进入 $\mathcal{J}^+$ 的比例不低于注册 stage | 显著低于 | 删自写来源，退回修复版 VERGE |
| P5 proposer 学到了东西：fine-tuned proposer 的新提议进入 $\mathcal{J}^+$ 的比例高于同一 archive 下的 frozen proposer | 无差异 | 学习只在 in-context，proposer 冻结 |
| P6 loop 有必要：matched-compute direct（生成 token 数相同）与 dense direct 不到 first success，loop 到 | direct 或 dense 先到 | 目标不在 regime 内，换目标 |
| P7 顺序有必要：fresh-start replay 中 discovered order 显著优于 pooled mixture | 无差异 | 只剩 stage 集合的价值 |

---

## 5. 执行顺序与算力

Gate 0（新 backbone regime，≈5 B200-h）→ Gate 1（可点燃性：注册族的人写 oracle、手写 rotate-until-marker 族、hinted-target oracle、dense direct、matched-compute direct，各 3 seeds，≈150 B200-h）→ Gate 2（单轮预测效度，≈40）→ Gate 3（闭环：本方法、SOAR reward、uncertainty 各 6 seeds，random / frozen / target-irrelevant 各 3，≈1,230）→ 其余。每轮算力与修复版相同：4 个 branch + 探针；分格与 hint 来源不增加 branch；hint-regret 探针每个新 stage 512 条 rollouts。

## 6. 稿子里可以写的 claim（范围内）

Solver 的 reward 自始至终只有可执行任务的 binary full pass；proposer 的 reward 是固定目标 verifier 最低未饱和必要条件上的 compute-matched gain，first success 之前非退化；分解只通过任务选择与 checkpoint 选择起作用，选择在按 rung 与结构分格的种群上进行；switch 之后 reward 化为 full-pass gain。其余全部是第 4 节的预测，按结果写。
