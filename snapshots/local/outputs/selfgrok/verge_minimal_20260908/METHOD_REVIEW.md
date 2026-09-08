# VERGE 最小真实实验复核

已完成含真实Solver参数更新的VERGE最小流程实验。经过两轮失败与定点修复，第三轮四分支、阶段探针、最终评价、评分和独立复算全部完成：g1发生一次binary-only LoRA更新，但三条课程Gamma均为0，所有目标采样均未出现full pass，J+为空。流程可执行；本次未观察到目标迁移收益，尚不能宣称方法有效。

本轮以用户提供的原始设计和实际工程为依据。原文逐字副本保留在相邻 `verge_review_20260908/evidence/`。用户后来明确授权补齐操作定义、必要时更换 benchmark，并要求实际完成最小实验；这覆盖交接文件中较窄的 CPU/smoke 执行边界。旧代码、初始 checkpoint、日志和其他作业均保留。

两个原文副本重新核验的SHA256：START为`36667fafbdf5df966391c1172fbfc6e3cf3d260e40a87ecbaf37e9f8832717cc`，own design为`8ecb336bdee144fd28a0b41f8bbdb8f8d29cac63cf2e4b12f43bd53c4b8a0442`。原文的预测不当成已有结果，旧会话意见不当成已批准定义。

## 最终实测

| 第三轮分支 | 生成训练token | 实际更新 | 最终目标full pass | Gamma（95% paired CI） |
|---|---:|---:|---:|---|
| g1：identity → target_small3 → prepend_RB | 65,536 | 1 | 0/256 | 0 [0,0] |
| g2：prepend_R → replace_RB_to_YG → target_small2 | 65,536 | 0 | 0/256 | 0 [0,0] |
| g3：replace_RB_to_YG → target_small3 → target_small2 | 65,536 | 0 | 0/256 | 0 [0,0] |
| direct | 65,536 | 0 | 0/256 | 对照 |

表中的0/256是各端点评价槽；后三个未变checkpoint共享原base证据，不能将其当成独立的额外768次采样。g1真正新增256条无hint目标评价、61,055token，原base的256条/60,151token跨轮复用。训练与全部阶段/最终评价记录均无target success；这个结论来自原始记录核对，不是把无全局时间顺序的first-success null当作零。

本轮kappa=2，所有stage Delta、target-only tail及P32/P8修正均为0；没有reward switch或learning onset。archive按原规则保留(2,1)格及原base lead，未产生绕路；J+为空，未做Q RFT。182个LoRA张量的实际变化已CPU直接比较确认，新的g1_m1 checkpoint仍单独保留，不能用它替代原规则选出的lead。

三轮及指定前置probe累计1,039,681生成token、2.5425 B200小时，指定新增作业峰值2 B200/128GiB。87项当前执行路径CPU测试、独立vendor重放、全部三轮原始评分复算通过。正式P1–P7与完整Gate0–3未验证。证据索引为`EXPERIMENT_RESULT.json`、`runs/round3_result/`及`evidence/COST_THREE_ROUNDS.md`。

## 当前判断

方法可以实现并运行到可核验结论。此前阻止训练点燃的一项工程问题是输出大量非法DSL。通用语法约束后的96个真实样本全部可解析，identity stage出现1/8完整成功，触发一次真实二值奖励LoRA更新。参数变化和独立新checkpoint已核实；这证明出现了可用梯度，不等于更新后的成功率已提高。

目标基线仍然困难：相同语法策略下，8个selection实例×32条采样，共256条均可解析，六条件率为 `[1,0,0,0,0,0]`。因此目前最弱测试类信号仍为零，原文“first success之前非退化”不能当作无条件保证。后续必须用课程/direct差异判断，不能以语法合法率、某个stage成功或参考程序可执行代替目标能力。

## 原文保留内容与本轮开发配置

- 研究问题保持为固定target下的课程/检查点选择；Solver仅学习当前任务的binary full pass，beta=0、wd=0，恒定奖励组精确跳过。
- 同backbone、独立Challenger/Solver LoRA；一个共享base，3条有序课程及1条direct，fresh optimizer，各65,536实际生成token。三阶段及target-only尾段各16,384，alpha=eta=.25。
- 保留全部三种stage来源、四项validator、min-per-class六条件、Gamma/reward switch/J+、分格archive、累计正例RFT、无hint目标评估、P1–P7及其原对照范围。
- 开发轮只用原selection顺序前8个实例；每题仍为36个测试、六类各6个，不改分母和阈值。阶段P8、最终P32固定槽，8个实例不是完整64实例的正式评估。
- 实际backbone为现场核实的Ministral-3-3B-Instruct-2512-BF16，两个初始adapter均来自当前VERGE工程；没有从其他项目推定模型或数据。

本轮继承用户放宽后已冻结的benchmark：输入RB、长度0–64，输出仍为`RBYGRB + input.replace("RB", "YG")`。六类按非重叠RB匹配数0/1/2plus与末尾是否R交叉划分；没有根据heldout/test模型结果选择分组。变更来源、新版Delta定义、尾段及P8/P32修正在相邻`verge_operational_20260908/METHOD_REVIEW.md`逐项登记，不能称为找回的旧稿。

## 此次新增工程选择的影响

通用grammar保留2–32节点、全部标准节点类型、任意循环及已声明引用；没有提供目标解、成功轨迹或固定拓扑。规范命名、声明顺序和受限采样会改变模型策略，必须对base、课程、direct和评估统一冻结。训练使用完全相同的grammar mask归一化native suffix log probability；不是将guided生成接回无约束loss。EOS计入预算，之后的batch padding不计。

Challenger本轮确实生成JSON，但stage规格选择被限制在9个冻结的可执行候选中。这是最小管线开发配置，不是开放变换合成已经有效的证据。已有受限表达接口继续保留；hint来源也保留，但没有完成512条两臂hint-regret探针的stage本轮不进入课程。N_min=8，未达到新增J+阈值时不进行RFT，不能声称P5已验证。

当前执行器处理独立的一轮，scorer每次从本轮base建立archive和buffer，过往结果另外进入Q context。持续多轮、多incumbent状态与跨seed buffer的生产接线尚未完成；RFT在这里输出执行计划，尚没有真实Q训练。相邻operational实现和CPU测试保留相关接口。这是本次最小实验的工程范围，不能表述为原文全部长期闭环已经实现；原研究问题与正式对照范围没有删除。

## 仍需实验回答的事项

min-per-class信号可能继续全零；archive是否保留实际有用的绕路、静态图环是否对应有效算法、课程能否迁移、顺序是否优于混合、Challenger是否跨seed学到东西，都仍是待测预测。本轮不改变这些问题，也不以一轮开发结果宣布P1–P7成立或被否证。一次可完成的实验与正向科研结论是两件需要分别核实的事。

## 前两轮结果与点燃修复的依据

`runs/round1_result/`已生成真实exchange、预算核算、Gamma/Delta、archive、事件和RFT状态。四条branch全程reward为0，optimizer精确跳过，权重与base一致；三条Gamma和含tail的各段Delta均为0。所有端点都有实际checkpoint与评价来源，这些零值不是缺失定义的替代。archive仍为(2,1)这一格，lead为原base，reward未switch，J+为空，RFT未触发。

256条真实base评价消耗33,854生成token。未更新checkpoint的16个后续评价alias复用同一证据，共覆盖2,048逻辑槽；这不代表新增了2,048条独立模型采样。四条实际训练共262,144token，全部保留。

进一步定点核查发现：80/96条无约束stage样本以数字0/1 token开头，而grammar只接受n0/n1等；原prompt没有说明这一限制。约束后84/96首跳为start/NONE。新测试只追加规范命名说明，保持相同任务、seed、cap、grammar和binary verifier，独立保留新旧版本。实际新版首跳start/NONE降为2/96，并出现一条append-R成功程序；2/96对1/96的成功数不构成显著效果证据。两版本192条程序/2848个测试均通过原vendor重放，详见evidence/NAMING_V2_RESULT.md。

命名版的第二轮也已完成：训练262,144token、0次课程更新、Gamma全部为0。新的256条base评价为60,151token，后续未变权重均准确复用。五组真实训练的40条程序/528个测试重放确认使用了新版concise提示，排除了“实际还在用旧提示”这个原因。

第三轮仅在g1第一阶段加入公开的人为提案先验，由真实Challenger在已测出mixed的identity/append_R中选择，其余八位置仍从同一菜单自选。真实输出选择了identity。没有代写Q输出、增加第五validator、改变初始化、喂入正确程序或提供target hint。这项干预改变开发提案采样空间，不能用来证明无约束Challenger已经自主发现有效课程；正式P1–P7范围保持待检验。第三轮复用第二轮同view、同权重、同完整sampling contract的原始base记录，新增baseline成本为0。

没有恢复CAFD的T1/T2、教师路径、KD/RL路由或MPC。
