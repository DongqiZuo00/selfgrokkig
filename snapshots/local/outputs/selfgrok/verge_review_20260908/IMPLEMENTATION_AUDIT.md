# VERGE 实现与运行审计
2026-09-08。真实工程根目录 R = `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness`。
远端独立交付目录 = `R/outputs/verge_review_20260908`；本地目录 = `C:/Users/ddong/Documents/Codex/2026-09-06/hipergator-x20/outputs/selfgrok/verge_review_20260908`。

**当前工程是旧 VERGE book/control/follow-on 与 target-only search；没有证据表明最新版 own-method 已完成迁移。** 本轮修改只在独立 `review_copy`；运行中源码与 checkpoint 没有被覆盖。

## 1. 来源、工程指令与冲突
完整读取 Downloads 中两份原文；Temp/Downloads 的 START 文档 SHA256 相同：
- START：36667fafbdf5df966391c1172fbfc6e3cf3d260e40a87ecbaf37e9f8832717cc
- own design：8ecb336bdee144fd28a0b41f8bbdb8f8d29cac63cf2e4b12f43bd53c4b8a0442

现场 Git：`master`，HEAD `b3298a61360f659b2a336decbd392252eb429e7c`，ahead 52。开始时已有 4 个跟踪文件变更，另有大量未跟踪 VERGE 源码/数据/成果；原状态和原 diff 已保存。审计结束 4 个跟踪变更仍相同，3 个本轮涉及的生产源文件 SHA256 与开始快照一致。

在本地工作目录祖先及个人 .codex、远端工程祖先及 self grok 内查找 AGENTS.md / AGENTS.override.md，未发现文件。没有可报告的这类文件正文冲突；**配置中确有旧科研约束，不因此成为本轮用户指令**：

| 现场来源 | 冲突/处理 |
|---|---|
| `manifests/verge_book_v2_protocol.json` source_document | 引用 VERGE_self_play_revised.pdf，不是本次 own_method_design。旧实验依据保留；未恢复旧稿整套设定。 |
| 同协议 explicit_user_overrides | 记录旧会话“一 seed、64GB、禁新 audit/hash scan、Mistral only”等。只作历史协议事实；本轮直接授权的核查继续，不继承科研结论或重训授权。 |
| `src/common.py`、`src/modeling.py` | 硬编码 Mistral 并拒绝 Qwen。记录现状，未改 backbone，也不把旧记录当作新版选择依据。 |
| `src/verge_search_protocol.py:validate_plan/validate_round_config` | 当前 target-only、无 Challenger、KL=.02、loss-token 预算被硬冻结。不能把这个入口当新版训练命令。 |
| 资源边界 | 用户前述 192GB/4 B200；START 文档写 2 B200；旧协议限制 2 workers×32GB。区别记录。本轮 GPU 使用为零；未来需冻结唯一资源协议，不能把上限当整套 Gate 的授权。 |

## 2. 模型、数据与现场状态（仅实现事实）
模型实际为 `mistralai/Ministral-3-3B-Instruct-2512-BF16`，位于 self grok/models 对应目录。配置架构 Mistral3ForConditionalGeneration；tokenizer 配置声明 LlamaTokenizerFast、无 chat_template，代码使用 Tekken/Mistral framing。模型配置、tokenizer 配置已记录 SHA256；读取的 manifest/config/adapter revision 均没有可核验 commit，定点下载 metadata 路径也未找到。因此**不能宣称精确 backbone/tokenizer revision 已锁定**。

book_v2 的两个根路径分别为：
`R/checkpoints/verge_book_v2_initial_solver/resume_u0000`
`R/checkpoints/verge_book_v2_initial_challenger/resume_u0000`。
两者 adapter 为 rank16、alpha32、dropout0，同本地 base；各有约 98.9MB safetensors。使用 `torch.load(weights_only=True,map_location="cpu")` 读取各 2.8KB state：trained_tokens=0、optimizer entries=0；lr 分别 1e-5/1e-6，Adam betas=(.9,.95)，wd=0。这是元数据与文件存在性证据，未新加载完整 backbone，未声称权重内容已逐位核验。后续实际选择路径见 field_evidence.json 中每轮 branches/decision；旧 Challenger 更新记录也在其中。

target 路径由 `src/verge_target_data.py:PARAMETERS/prepare_target` 和现有 manifest 指定：`prepend_sequence`，四色，replace_pattern_to_pattern，RBYGRB / RB / YG。现有 wrapper 行为与目标断言一致；不从别的项目引入 target。
数据为 `R/data/verge_mistral_round1/{target_train,target_selection,target_completion}`：
| 划分 | 行数 | 每行测试数 | 任务参数种类 | 唯一输入 tape 数 |
|---|---:|---|---:|---:|
| target_train | 128 | 19–23 | 1 | 384 |
| target_selection | 64 | 19–23 | 1 | 215 |
| target_completion | 64 | 19–23 | 1 | 213 |

三对划分 ID 交集、完整 test-suite 交集均为 0；输入 tape 交集依次为 50（train/selection）、55（train/completion）、43（selection/completion），任务参数完全相同。若“实例参数元组”指任务参数，则不满足字面不相交；若包含生成 seed/测试集合，则需原文明确。**这些重叠不自行判成泄漏，也不能宣称 held-out task-family 泛化。** completion 不等于官方 sealed test。原目标行无 hint 文本；真正的 held-out/test 模型结果未读取。

## 3. 测试类和 Delta 的定点核查
用户已明确这两项是未闭合依赖，不能补猜。以下路径相对 `/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe`：
- `manufactoria/manufactoria_problem_generators/prepend_sequence.py:26`，PrependSequenceGenerator；`_get_pattern_specific_cases` 提供若干长度/模式样例，`generate_problem` 合并、去重并生成平坦 test_cases。
- `manufactoria/manufactoria_problem_generators/base.py:98`，BaseGenerator._generate_deterministic_test_cases：拼接 basic、corner、pattern-specific，再按输入去重，只返回 List[str]。**这是生成步骤，不是分类映射；重叠来源在去重后不可据此恢复。**
- `manufactoria/hf_file_wrapper.py:176`，TrainingFileWrapper._extract_test_cases：仅保留 input/expected_output/expected_accepted/check_output/description。
- `R/src/verge_round_core.py:condition_vector` 用全部 cases 分母，没有 class 逻辑；现有数据字段、配置、manifest 未找到 test_class/class_id/class_mapping。
- 全部目标测试都有 flat-list 位置，但没有正式各类名称、归属、分母、空类策略或完整类别覆盖证据。没有把 description、长度、mutation 当分类依据。

Delta 检索覆盖 R/src 的相关 VERGE 代码/测试、配置/manifest、git log -G 与已跟踪路径；未找到 per-stage/telescoping 候选。出现的 `verge_repair_training.py` delta 是 KL log-prob 差；`verge_control_analysis.py` delta 是端点矩阵差，均不对应课程逐阶段信用。
因此 stage 前后 checkpoint、target tail、固定 reward condition、direct 如何进入、阶段/最终采样衔接、telescoping 端点六项均 **MISSING_DEFINITION**。当前 stage_probe_instances=0 也无法追溯新版分解；缺失不是零收益。

## 4. 逐组件对照
下列路径均相对 R；具体文件已随 review_copy 提供。
| 组件 | 真实路径/符号 | 与当前原文对照 |
|---|---|---|
| per-test verifier | `src/common.py:verify_full/verify_all_tests`；vendor RobotFactory.process_robot | 官方 DSL 解释执行，1000 步上限；无执行任意 Python proposal。verify_full 首失败即返回，verify_all_tests 才逐项继续。实际目标 all-accepting exact 输出；通用 recognition/regex 行为不是新版错误语义的自动批准。 |
| 条件/LCP | `src/verge_round_core.py:138–183` | 11 项旧 ladder：parse、execute-all、累计 LCP 阈值、累计整体 fraction、full。LCP 进入选择/Challenger reward；与新版相冲突。未偷偷改成 6 维或一类 f。 |
| stage/validator | `src/verge_repair_protocol.py:proposal_schema/parse_typed_proposal`、`verge_repair_prepare.py:stage_rows` | 注册族+受限 finite_map。无完整三类 stage/hint 库和四项 validator；常量 finite_map 可通过 schema。不能把 schema 合法视为 validator 合格。 |
| shared base/optimizer | `verge_round_prepare.py`、`verge_repair_train.py:main`、`verge_round_runtime.py:optimizer_for` | 旧 curricula/direct 共用 cfg.solver_start；新 branch fresh，resume 恢复自己状态。不是 Challenger 从多格选择 base。beta 在这里是 KL 系数 .02；Adam betas 另为 (.9,.95)，selector_beta=2 又是另一参数。 |
| binary reward/skip | `verge_repair_train.py`、`verge_repair_training.py:train_step` | 课程和 target 分组验 reward∈{0,1}，零 replay；曾经更新后即使 beta=0 也会继续 AdamW，已在副本修复。非零 KL 旧行为保留并明确不符合新版 beta=0。 |
| B/alpha/eta | `verge_round_core.py:stage_tokens`、`verge_repair_protocol.py:prompt_role`、`verge_repair_train.py` | 旧 524288 是 loss tokens，phase 边界 mask，生成 surplus 另计；alpha/eta=.25，target-only search=1。不能声称 generated-token matched。 |
| 阶段/最终探针 | `verge_repair_config.py`、`verge_round_runtime.py:endpoint` | 当前阶段探针关闭，64×8 selection；不是新版每 stage 8、最终每题32。无 hinted target 的实际链路可测。 |
| Gamma/switch/J+ | `verge_round_decision.py:paired_interval/decide` | reward 切换需≥2 branches，测试已核实；但 selection≥1 即切换，并加 scope guard、softmax 选择。没有独立按 reward kappa 的 J+ RFT 流程。旧 CI 以 instance 外层、配对 rollout 内层重采样；不代表训练 seeds。 |
| archive/lead | `verge_book_protocol.py:make_config`、`verge_book_runtime.py:history` | 累积历史 JSON + 单条 selected_solver 延续；没有 (b,s) incumbent grid、含环率、原文 lead。保存了其他 checkpoint 文件不等于实现多起点 archive。 |
| Challenger | `verge_repair_teacher.py:update_teacher` | schema-normalized 在线 policy-gradient，一轮一次，signed gains，跨轮继承 optimizer；与累计 J+ / N_min / 一遍 RFT 冲突。未发现 N_min 触发器。 |
| 事件/结束 | `verge_search_events.py`、`verge_book_protocol.py` | 有旧 training/endpoint first-success 信息；没有完整新版格占用、非 lead base、learning onset 体系；旧协议强制六轮且不早停。 |

## 5. 运行结果核验及边界
field_evidence.json：24 轮（4 arms×6）96 个 endpoint 的 11维 counts 从保存的 keyed_vectors 全部重算一致，full successes 总计 0/49152；训练 target successes 总计亦为 0。这个分母是重复/相关端点采样的总和，不能作 49152 次独立训练或用来给总体零概率结论。
旧四臂 loss tokens 各 12,582,912；generated tokens 分别：
VERGE 12,863,846；frozen 12,789,221；uncertainty 12,752,651；outcome 12,791,270。训练预算也不能改名为等生成预算。
旧 VERGE Challenger 更新 5/6，uncertainty 6/6，frozen/outcome 0/6；与新版 J+ 机制无关。

进一步按预选规则，重放 book_v2_verge_r02 direct/candidate1 各前32条原始 target completion，64/64 binary reward 一致；不是随机总体抽样审计，不涵盖所有旧原始输出。
PRIMARY_BLOCK_COMPLETE 明确 suite_complete=false。18 个 direct segments 与 72 个 follow-on 的完成标记存在，本轮只核对其标记/范围，未逐条重放这些运行，不能替代新设计 Gate。
初次现场是 41360248 prepare RUNNING、41360249/41360250 PENDING；后续 prepare 结束，41360249_0/_1 RUNNING，其余/finish 等待。查询已结束 prepare 的 scontrol 返回 invalid job id，未据此判作失败。它们是已有旧 target-only 作业；本轮没有提交/取消/接管。每个 worker 脚本请求 1 B200、32GB，数组并发2。

## 6. 实际最小修复、测试及未迁移范围
修改仅在独立 review_copy：
1. `src/verge_repair_training.py`：beta=0 且无 active advantage 时，不再因 prior_steps>0 调用 optimizer/scheduler；保留非零 KL 旧分支。实际 Tiny CPU AdamW：修复前参数最大变化 0.0033932384，修复后 0，optimizer moments/step counters 与 scheduler 字典精确相同。全0、全1、fresh 和 warmed optimizer 均覆盖。
2. `src/verge_round_core.py`：profile 拒绝空样本、重复 instance/rollout、漏实例/不一致 rollout 集合，不再静默覆盖和变分母；阈值拒绝非正整数；有效既有 profile 计数不变。
3. `src/common.py`：verify_full/verify_all_tests 在解析前拒绝空测试集、缺失/非字符串 expected_output；显式空字符串仍合法，check_output=False 的合法 recognition 接口保留。condition_vector 同样先检查。wrapper 在序列化前已默认填空的历史输入仍无法由此识别；未宣称补齐全部数据契约。

未修改旧 11维 ladder、目标划分、class 映射、archive、Challenger 算法、训练预算或正在运行配置。完整新版迁移需要版本化实现，不能靠这3处修复完成。

测试日志：
| 记录 | 结果/解释 |
|---|---|
| `evidence/baseline_all_regressions.log` | 新增11个方法在原快照上出现8个失败（含 subtest）和1个 error；保存复现，不丢弃失败。 |
| `evidence/cpu_regressions_release.log` | unittest 共68个测试执行通过，含新增11项及相关旧协议回归。 |
| `evidence/spec_fixtures.log` | 10项合成规格测试通过：显式 class 聚合/固定分母、Gamma、至少两分支 switch、J+、archive 空格/替换/lead、hint 反例。**不调用正式新版组件；不得称其已实现。** |
| `evidence/contract_diagnostics.json` | 真实 parser 1000步超时 fixture 为 [0,1,1]；旧 ladder 会压成 [1,0,…,0]，证明与新规格不同；包含 optimizer 和64条重放证据。 |
| `evidence/cpu_regressions.log`、`cpu_regressions_with_vendor.log` | 早期隔离环境分别缺 manufactoria / hf_file_wrapper 路径；补齐同一 vendor 包的两级 PYTHONPATH 后通过，未改生产代码解决环境问题。 |

运行命令和新脚本见 RUN_CPU_TESTS.sh、evidence/command_log.json、evidence/spec_fixtures.py。
baseline 与 diff 为同次现场快照生成；`patches/minimal_fixes.patch` 可审阅，未对旧实验自动应用。无 GPU smoke、无完整模型加载、无新增训练或 Gate 0–3。

