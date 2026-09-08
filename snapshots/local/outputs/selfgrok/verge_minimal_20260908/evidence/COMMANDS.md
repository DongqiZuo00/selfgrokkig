# 本轮实际命令索引

远端工作目录：`/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908`。命令通过已有 HiPerGator SSH alias 执行。所有产物位于 selfgrok；没有提交完整 Gate 0–3。

已经执行的 GPU 提交：

```bash
bash SUBMIT_PILOT.sh runtime/probe_stages.py stage_screen_retry1
bash SUBMIT_PILOT.sh runtime/probe_grammar.py stage_grammar_v1
bash SUBMIT_PILOT.sh runtime/prepare_round.py round_prepare
bash SUBMIT_ROUND1.sh
bash SUBMIT_NAMING_V2.sh
bash SUBMIT_PILOT.sh runtime/prepare_named_round.py round_prepare_named
bash SUBMIT_ROUND2.sh
bash SUBMIT_PILOT.sh runtime/prepare_ignition_round.py round_prepare_ignition
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/preflight_ignition.py --preparation runs/round_prepare_ignition_41419999
bash SUBMIT_ROUND3.sh '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/runs/round_prepare_ignition_41419999'
```

对应实际 job：41410864、41412533、41413994、41414619、41415498、41417194、41418169。首个错误启动 41410512 另外保留，未将其计为有效采样。各 job 的标准输出/错误见 `runs/<label>_<job>.out/.err`，数组为 `runs/roundN_<array>_<task>.out/.err`；raw、JSON journal、fragment 与 COMPLETE 分别保存在独立 run 目录。

第三轮实际 job 为41419999和41420288。提交前运行`prepare_ignition_round.py --dry-run`，证据在`runs/ignition_cpu_preflight_complete/INPUTS.json`，所有依赖已验证；第一stage先验与baseline复用证据均经过真实CPU检查。第三轮合同为`configs/round3_runtime_contract.json`，SHA256 `cd7a922f0effbfa49e4bea1ac77c7bf2aa0151e1318953530f4295ac9e3b85b8`。

CPU 检查与结果处理：

```bash
module load python/3.11
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/preflight_round1.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/finalize_round1.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/preflight_round2.py
python3 runtime/round2_results.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/round2_results.py --finalize
python3 runtime/audit_real_scores.py --preparation runs/round_prepare_41413994 --result runs/round1_result --output evidence/round1_independent_score_audit.json
python3 runtime/audit_real_scores.py --preparation runs/round_prepare_named_41417194 --result runs/round2_result --output evidence/round2_independent_score_audit.json
python3 runtime/round3_results.py --compact
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/round3_results.py --finalize
python3 runtime/audit_real_scores.py --preparation runs/round_prepare_ignition_41419999 --result runs/round3_result --output evidence/round3_independent_score_audit.json
bash COLLECT_ROUND3.sh
bash evidence/naming_v2/RUN_REPLAY_CPU.sh
bash evidence/round3_update/RUN_REPLAY_CPU.sh
```

第二轮第一次 preflight 因一个启动器日志字段未同步而失败；diff 和旧文件完整保留在 `ROUND2_DEPLOYMENT.md` 所列路径，同步合同要求的版本后通过，没有绕过校验。

完整 CPU 命令和日志还分别保存在 `benchmarks/generated/`、`decoding/`、`round/fixtures/`、`runtime/*cpu_tests.log`、`evidence/naming_v2/`、`evidence/round2_verifier/` 和 `evidence/ignition_schema_cpu_20260908/`。其中 `SYNTHETIC_GRAMMAR_ONLY_*` 明确是 schema fixture，不是 Challenger 实际输出。

`runtime/audit_real_scores.py` 不调用正式 scorer，直接以保存的原始 per-test binary 输出和冻结 class 映射重算条件、Gamma、tail Delta、P32/P8 修正；它另核验权重、raw SHA、native token IDs 与缓存来源。它不会重采样模型，vendor 重放由另外的 verifier 审计完成。

本地最终验收已实际执行：`C:/Users/ddong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B outputs/selfgrok/verge_minimal_20260908/runtime/final_acceptance.py`，PASS日志为`evidence/FINAL_ACCEPTANCE.log`。三轮结果与新checkpoint镜像均已核对，所有新增Slurm作业终结。验收脚本写入新的`EXPERIMENT_RESULT.json`，重复验收须使用独立输出版本，不能覆盖已交付原件。
