# 真实单轮交换协议与结果汇总

本目录只负责计划冻结、读取真实 rollout/训练 hash journal、计分和生成报告。它不导入 Torch、不加载模型、不生成完成文本、不提交作业，也不构造任何科研结果。模型后端必须另行产生原始记录。`fixtures/` 专用于单元测试；正常 CLI 不提供 fixture 开关。

本次开发 mini 保留 G=3 个有序课程及一个 direct，所有 branch 从同一 base 权重与 fresh optimizer 开始，三个 stage 加 target-only tail，Solver 始终 binary full pass、beta=0、weight_decay=0。默认 **B=65536 generated tokens/branch**、eta=.25、alpha=.25，四段各 16384，共 262144 训练 tokens。相较 B16384 的每段4096，这允许完整八条组在段起点使用每条2048 cap。较小 B16384/32768 可显式选用，实际值入计划，绝不按 loss-mask 冒充 generated budget。

使用固定 **8 个 selection 实例**完成这轮开发验证；每个实例仍保留原来的完整36测试、6类及同一目标语义。它不等于正式64实例设置，没有删除研究scope或将单轮开发结果当作任何P1–P7的证明。所有 base/direct/curriculum 探针与最终评价使用同一无解、无 hint、冻结 hash 的 target prompt view；改变的是文字说明压缩，不是语义或测试。中间 probe 为每题8条，final为每题32条，最终P8直接取同份P32原始记录的前8个采样槽。

## 后端调用

```text
python -B round/round_cli.py plan --request request.json --output plan.json
python -B round/round_cli.py score --plan plan.json --exchange exchange.json --output result_directory
```

也可 `from round_cli import build_plan, score_exchange, sha, file_sha, generation_seed_for_sample`。输出采用新路径独占创建，避免覆盖旧结果。代码复用同级 `verge_operational_20260908/{protocol,runtime,stages}`，因此部署时保留两个相邻发布目录。

`request.json` 必须提供：

- `round_id`、整数 `training_seed`、实际 `base_checkpoint`。
- `selection={manifest_id,instance_ids:[8],sample_ids:[32],prompt_view_sha256,test_contracts}`。`test_contracts[instance_id]={test_ids,class_by_test,expected_classes}`，由冻结 ground_truth/manifest 导出，不从模型结果反推。
- `proposal={base_checkpoint,curricula:{g1:[stage,stage,stage],g2:[...],g3:[...]}}`；每个 stage 恰为 `{stage_id,kind,spec}`，kind 支持 registered、tape_transform、hinted_target。
- `challenger={checkpoint,raw_output,context,evidence_id}`；raw_output 必须是实际生成的JSON字符串，解析后与 proposal 完全相等。这里不能事后用手写课程替换模型提议而仍声称 Challenger 生成。
- `stage_validations[stage_id]`：真实 validator 的 `accepted/checks/details`，并附 `spec_sha256=sha(stage.spec)`、`evidence_id`。无 hint 的第四项为 `NA`；hinted_target 还必须有已验证train程序的 `details.hint_provenance`、实际通过的512条两臂 `details.hint_probe` 和 `hint_probe_evidence_id`。库为空或probe缺失就不能入计划。可以一轮全用非hint stage，但用 `stage_source_status` 明确记录原因，不删除第三种schema。
- 可选 `B`、`eta`、`alpha`、`q`（默认4）、`n_min`（默认2）、`bootstrap_resamples`（默认2000）。

plan 返回 `branches[branch_id].phases`，每段有 `phase_id/quota/tokens_before/tokens_after/target_probability/stage_id/checkpoint_alias`。共有17项 `evaluation_requests`，每项都有 **alias、checkpoint_alias、branch、milestone、tokens、n、sample_ids、hint:null**。base alias为 `base`；中间为 `g1_m1`..`g1_m3`；最终evaluation alias为 `g1_final`，对应checkpoint alias `g1_m4`。direct同理。`generation_seed_for_sample` 按 training seed、selection manifest、instance及sample slot生成公共随机种子，不使用checkpoint身份。

## 真实 exchange

顶层为 `{schema_version:'verge-mini-round-exchange-v1',mode:'real_model_round',plan_sha256,checkpoints,evaluations,branches}`。

- `checkpoints[alias]={path,weights_sha256,weights_file?}`。默认真实权重文件为 `path/adapter_model.safetensors`；score重新算hash。多个alias可以指向同一未改变的实际checkpoint，不能捏造保存路径。
- `branches[id]={base_checkpoint,fresh_optimizer:true,optimizer_initial_state_entries:0,beta:0,weight_decay:0,phases:[4个PhaseLedger asdict],journal_path,journal_sha256}`。真实后端用原 `generated_budget.run_phase` 生成完整 hash 链 journal；phase name须等于计划phase_id。score逐条核算实际token、binary reward、mixed/constant组及drain，核对提交ledger、四段预算与direct/tail target-only。没有optimizer step的段必须保持相同权重hash。报告同时给出mixed optimizer steps与确有权重改变的课程段，不能把恒定reward跳过称作训练更新。
- `evaluations[alias]={checkpoint,records:[...]}`。每条 record 为原 `RawRollout` 的七字段：`checkpoint,training_seed,manifest_id,instance_id,sample_id,conditions:[6个bool],has_cycle:bool或null`；另外必须有 `raw_completion,completion_token_ids,hint:null,generation_evidence_id,prompt_view_sha256,test_outcomes:{test_id:bool或null}`。score由完整per-test结果重新计算全部条件，不接收单独一个汇总通过率代替原始分母。
- 记录可附 `global_rollout_index` 标注真实采样顺序。该顺序只用于“提供的无hint selection评价流中的first success”；各branch训练target首次通过另从其journal记录，跨branch全局训练顺序若未给出就不虚构。

精确评价缓存：`evaluations[alias]={checkpoint:实际原路径,evaluation_cache_hit:true,source_evidence_id:'base'}` 可以省略records；scorer按source alias依赖解析原始记录并取本次sample槽，source在JSON中的先后顺序不重要。若提供records则核验与source完全一致；checkpoint必须是实际相同路径，所有manifest/instance/slot仍固定。未更新direct可共享base P32及其P8。若stage3已有P8、tail无更新，final可用 `evaluation_cache_hit:false,source_evidence_id:'g1_m3',records:完整P32`，其中前8必须逐条等于source，只新增24条；缓存依赖环会报错。报告区分逻辑覆盖槽数、cache命中和真正新生成的rollouts/tokens，不把重复引用算成新样本。

## 输出与检验边界

score一次生成 `summary.json`、`round_score.json`、`delta.json`、`events.json`、`archive.json`、`budget_audit.json`、`evaluation_provenance.json`、`lineage.json`、`challenger_buffer.json`、`rft_plan.json` 和输入来源hash。Gamma、reward switch、paired bootstrap、J+及分格archive复用已有新协议。Delta固定本轮kappa、包含target tail、使用同预算direct端点，并保留P8到P32 sampling correction；不会零填缺失端点。

全轮经验与per-stage Delta进入archive；只有J+进Challenger正例buffer。满足新增Nmin只生成累计单epoch RFT计划，**不会假称已经训练Challenger**。真实RFT的checkpoint与训练日志由后端另行执行、登记；J+为空则保留空正例事实和全轮经验。

14项单元测试全部使用 `fixtures/fixture_round.py` 生成的显式合成输入，覆盖真实接口的计划、缺失hint/raw拒绝、per-class聚合、预算链、Gamma/Delta、archive/RFT、完整/部分/前向依赖cache与fixture隔离，日志为 `fixtures/cpu_tests.log`。最初一项测试对异常措辞期待不匹配，已修正测试断言，原始日志保留为 `fixtures/cpu_tests_initial.log`。这些测试不是模型实验。
