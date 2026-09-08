# 实现与真实运行审计

独立远端目录：`/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908`。本地镜像：`C:/Users/ddong/Documents/Codex/2026-09-06/hipergator-x20/outputs/selfgrok/verge_minimal_20260908`。原仓库HEAD实测为`b3298a61360f659b2a336decbd392252eb429e7c`，本轮实现均写入独立outputs目录。

## 已完成的真实工作

| 作业 | 行为 | 实测结果 |
|---|---|---|
| 41410512 | 第一份stage启动 | 本地`$PWD`误展开到Slurm路径，0秒启动失败；保留记录，改为上传的远端提交脚本 |
| 41410864 | 原生无约束stage筛查 | 96条，parse 1/96，binary success 0/96，62,977生成token，0次更新；340秒、1 B200 |
| 41412533 | 通用DSL grammar stage筛查 | 96条，parse 96/96，binary success 1/96，10,089生成token；identity 1/8触发1次真实masked LoRA更新；194秒、1 B200 |
| 41413994 | 初始base无hint评价、独立Q提案 | base为256条，六条件率`[1,0,0,0,0,0]`；真实Q JSON及冻结plan已生成 |
| 41414619 | 四分支数组、并发2 | 全部完成，训练共262,144token，0次更新，三条Gamma=0，J+空；结果见`runs/round1_result/` |
| 41415498 | 仅追加命名说明的96条配对筛查 | 已完成，96条均parse，2条binary success、2个mixed组，17,031生成token；identity发生1次真实更新，append_R出现成功但该probe未更新它 |
| 41417194 | 命名修正后的base与Q | 已完成530秒；256条base评价消耗60,151token，实际Q生成2,726token，仍选择较难课程 |
| 41418169 | 第二轮四分支对照 | 全部完成，各65,536token、0更新、三Gamma=0；实际耗时453/475/488/475秒 |
| 41419999 | 条件点燃提案 | 已完成147秒；真实Q选择g1从identity起步；相同baseline原始记录复用，新增baseline=0 |
| 41420288 | 第三轮四分支对照 | 全部完成，各65,536训练token；g1实际1次更新，其余0；新增g1无hint评价256条/61,055token。耗时865/489/490/472秒，均exit0:0 |

真实stage采样、逐测试重放和更新证据见`evidence/stage_ignition/`及`evidence/STAGE_IGNITION.md`。identity成功程序是START直接到end，保留输入，完整通过16个训练侧测试。它没有解决目标变换。

## 源码接线与实际修改

| 组件 | 本目录源码/证据 | 核查要点 |
|---|---|---|
| stage候选与目标view | `benchmarks/stage_catalogue.py`、`freeze_pilot.py`、`generated/pilot_view_v1/` | 九种CPU验证候选；128训练行、selection前8行；仅压缩target messages，36测试与六类映射不变 |
| Solver模型接线 | `runtime/hf_backend.py` | 现场模型/adapter验证、native token生成、完整binary verifier、fresh optimizer、constant skip、masked真实梯度、独立checkpoint |
| DSL/JSON解码 | `decoding/dsl_grammar.py`、`grammar_service.py`、`hf_grammar_bridge.py` | 现有vllm环境CPU xgrammar，训练保留uncertainty环境；同一mask生成和loss，EOS/pad实测 |
| preproposal base与Q | `runtime/prepare_round.py`、`prepare_named_round.py`、`prepare_ignition_round.py` | 第一轮base先测；第二轮用新view重新测base；第三轮核验并复用同view base。三个Q均由独立LoRA实际生成JSON，所有attempt先存raw |
| generated预算与阶段执行 | `runtime/execute_round.py`，复用相邻release的`generated_budget.py` | 四段真实生成配额、完整8条组、边界drain、模型变化证明、target-only direct/tail、阶段checkpoint |
| 评价与缓存 | `runtime/execute_round.py` | 同实例/chunk随机种子，P8为P32前8槽；同一未变权重的已有raw可复用，新增成本单列 |
| Gamma/Delta/archive | `round/round_cli.py` | 全部测试分母、固定kappa、tail与sampling correction、paired CI、J+、分格archive、RFT阈值 |
| 启动与源文件冻结 | `configs/round1_runtime_contract.json`、`round2_runtime_contract.json`、`round3_runtime_contract.json`及各自Slurm入口 | 三轮分别核验合同列出的10/16/20项文件；单作业1 B200/64G/1h，新作业最多并发2 |

改动没有覆盖原工程代码。第一轮来源包为本地`work/selfgrok/verge_minimal_round_bundle.tar.gz`，SHA256 `990db3526b32bf725de8dd92d14434793bf72a84637ebd3eda0518a53c5fb5be`。第一轮prepare加载之后本地新增的READY字段不冒充该作业使用过的代码；wrapper兼容真实旧READY，并独立核验该轮合同。

第二轮prepare包为`work/selfgrok/verge_named_prepare_bundle.tar.gz`，SHA256 `bc2bbdbfd5186b321bf8f6371120b2b812142e3946fa0643c0a436e00f428bcc`；启动包为`work/selfgrok/verge_named_branch_bundle.tar.gz`，SHA256 `7192fe5825e520fd2f8d6d0023080e2dad1c8c02703768369c9ec6715b3c50cd`。另有一次已记录的wrapper单行日志字段同步，前后diff见`evidence/ROUND2_DEPLOYMENT.md`，通过round2合同后才提交。

第三轮来源包为`work/selfgrok/verge_ignition_round_bundle.tar.gz`，SHA256 `dbd0485bd94398f5f2d8041c516eb391e639d882fa6de1c7a48e0b1490066365`。它包含新的ignition入口、preflight和提交脚本；公共backend/executor/scorer维持冻结版本。第三轮实际合同SHA256为`cd7a922f0effbfa49e4bea1ac77c7bf2aa0151e1318953530f4295ac9e3b85b8`，READY与每个LAUNCH都指向同一真实prepare41419999，不能将第一轮包当成三轮共同来源。

## 测试与范围

59项首轮新增CPU测试分别为stage catalogue 11、prompt view 7、DSL/JSON 14、round scorer 14、executor 8、branch wrapper 5；其后两个命名view各4项、named prepare 10项、ignition prepare 10项亦已通过，当前执行路径共87项。另有未采用的数字命名grammar独立6项CPU测试；成本审计工具v3有18项（含v2的12项），不混入训练路径测试。标注fixture的测试只证明接口和计算逻辑；真实tokenizer/worker重放也不当成模型采样。另有真实输入合同预检、真实输出vendor重放、GPU实际mixed更新和已完成的匹配对照。

日志位置包括`benchmarks/generated/*log`、`benchmarks/generated/pilot_view_v1/*log`、`decoding/cpu_*`、`round/fixtures/cpu_tests.log`、`runtime/execute_round_cpu_tests.log`、`runtime/run_branch_job_cpu_tests.log`、`runs/round1_cpu_preflight.json`。早期测试失败和启动失败均保留，不抹除失败历史。

已实际使用的关键命令，须在以上远端目录执行：

```bash
bash SUBMIT_PILOT.sh runtime/probe_stages.py stage_screen_retry1
bash SUBMIT_PILOT.sh runtime/probe_grammar.py stage_grammar_v1
bash SUBMIT_PILOT.sh runtime/prepare_round.py round_prepare
module load python/3.11
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/preflight_round1.py
bash SUBMIT_ROUND1.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/finalize_round1.py
bash SUBMIT_NAMING_V2.sh
```

这些提交已发生，提交脚本会拒绝同名重复运行。`runtime/inspect_round.py`只读取当前真实日志，可重复使用。正式Gate0–3全套、多seed效果比较、hint-regret及RFT未因这些命令自动启动。

四分支全部原始记录已同步到本地相同`runs/`路径；收集包`work/selfgrok/round1_core.tar.gz`的SHA256为`2d63552897cad6a65e67181f0cd54be1255adc240b84ca12a5d2376543bcc8d5`。16个未变权重的评价alias共享base证据，实际新评价256条/33,854token，逻辑覆盖2,048槽。不能把缓存当额外独立样本。

第二轮同样全部保存并同步，包`work/selfgrok/round2_core.tar.gz`的SHA256为`8a15c767a20ef1671c6fa5d4baf23e6edd4322f87bb8a113de3f795ecd0a3e0c`。原始逐测试评分独立复核见`evidence/round1_independent_score_audit.json`和`round2_independent_score_audit.json`，两轮均PASS，检查全部36测试/六个非空类、checkpoint权重、原始native IDs、缓存、Gamma及含tail的Delta。训练提示/vendor实查见`evidence/ROUND2_VERIFIER_REPLAY.md`。

前两轮及指定前置probe实际成本为713,920 native tokens、1.858333 B200小时，来源647个原始生成文件。范围不含前一release的smoke或其他项目；独立报告`evidence/COST_AND_BUDGET.md`保留该截止范围。第三轮新增成本另算，同view的原baseline不会重复计费。

第三轮所有原始训练/评价/评分已同步本地，收集包`work/selfgrok/round3_core.tar.gz` SHA256为`b4394bae0f871d54346d2bcc390c1504d595a301ceaa3c826fe060180100e5fc`。唯一更新checkpoint的完整权重也已备份到本地`runs/round3_g1_41420288_0/execution/checkpoints/g1_m1/`，SHA256为`b5278b1fed5e784945b6770b9abfd3b8d0cede946d2d98ffd771871f2fc86090`，与远端完全一致。

`evidence/ROUND3_UPDATE_REPLAY.md`直接CPU比较364个LoRA张量，182个张量、12,779,520个元素发生变化，原base未变；mixed组及随后constant组共16条程序/416次测试重放通过。`evidence/round3_independent_score_audit.json`从64个真实原始batch重算全部2,048逻辑评价槽、六类条件、Gamma及含tail的Delta，PASS。结果三Gamma=0、J+空、archive1格、lead原base、未发生target成功或Q RFT；没有把stage成功当target收益。

最终累计成本见`evidence/COST_THREE_ROUNDS.md/json`：1,039,681 native tokens，其中训练786,432、Solver评价155,060、Q8,092、stage probe90,097；Slurm分配2.5425 B200小时，指定新增作业峰值2 B200/128GiB。第三轮summary引用512条/121,206token评价证据，其中本轮新增仅256条/61,055token，原base复用部分未再次计费。最后验收见`EXPERIMENT_RESULT.json`与`evidence/FINAL_ACCEPTANCE.log`。本轮新增作业已全部退出，旧作业不受影响。

## 原文、实现事实与未覆盖事项

原文规定的奖励、课程/direct、archive、Q角色和预测范围保持。用户授权后补齐的类别与Delta是新操作定义；grammar与有限catalogue是本轮开发配置。旧工程的11条件、KL=.02、loss-token B、单血统、在线Challenger策略梯度没有进入本轮。未发现适用AGENTS文件；交接文件两卡上限与用户直接4卡/192G指令的冲突以用户指令为准。

本轮尚不提供开放变换生成有效性、hint迁移、Challenger RFT学习或P1–P7的完整证据。当前`pilot_state.generated_tokens`含探针，训练预算必须读phase ledgers；原raw的adapter字段标识加载来源，训练后状态须结合optimizer步数、checkpoint fingerprint和hash链解释。

须区分“未运行”与“未完成生产接线”：`round/round_cli.py:score_exchange`新建CheckpointArchive和RejectionSamplingBuffer，一轮输出实际更新后的状态及RFT plan；当前入口没有将多轮incumbents/buffer继续传回执行器，也没有执行Q RFT。历史作为真实context输入已运行，但这不等于完整持久闭环。开放stage表达、hint来源和跨seed正例接口仍在相邻operational版本，当前有限菜单/首阶段先验不验证这些机制的有效性。
