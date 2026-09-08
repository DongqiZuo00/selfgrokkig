# Selfgrok 本地归档盘点

盘点时间 UTC：2026-09-08T23:15:51.830983+00:00。未提交、上传或访问远端文件系统。

两个源目录合计 **2,752 个文件，266,937,717 bytes（254.57 MiB）**。本归档准备目录自身排除，避免递归把盘点结果计入源数据。压缩包按磁盘 bytes 计一次；内部成员另列，不叠加计数。

| Release / 来源 | 文件数 | Bytes | MiB |
|---|---:|---:|---:|
| outputs/selfgrok/verge_minimal_20260908 | 1,447 | 234,882,920 | 224.00 |
| outputs/selfgrok/verge_operational_20260908 | 304 | 13,704,070 | 13.07 |
| outputs/selfgrok/verge_review_20260908 | 511 | 3,277,890 | 3.13 |
| work/selfgrok/[root files] | 23 | 12,930,608 | 12.33 |
| work/selfgrok/baseline | 467 | 2,142,229 | 2.04 |

## 包含范围

- 旧 VERGE：review release 的 review_copy 中包含 book v1/v2、control、followon、search 等历史 manifest 与工程副本；仅盘点文件，不重新判断旧科研结论。
- 当前 VERGE：operational 协议/CPU 测试与 minimal 的三轮实验、诊断、预算报告、训练 journal、raw 证据、runtime contracts 及第三轮 g1 更新 adapter。
- work/selfgrok：baseline 工程副本及传输 tar.gz；baseline 也含其他旧 self_grok/recipe 脚本，不能把所有文件都标作当前 VERGE 实验。

## 原始来源与远端权重镜像

三轮及前置 stage probes 的成本报告列出 876 个 raw 来源；按内容 SHA 检查：{"content_identical_alias_or_archive": 36, "exact_mirror": 840}。原路径缺失但其他位置内容 SHA 一致的文件不判为数据丢失。

本地权重文件数：1。远端 checkpoint weight 引用中有 4 个路径在约定镜像位置未发现。它们来自本地 manifest/运行证据，未据此断言远端现在仍存在，也未把历史输出声明当成已经下载的权重。

| 本地权重 | Bytes | SHA256 |
|---|---:|---|
| outputs/selfgrok/verge_minimal_20260908/runs/round3_g1_41420288_0/execution/checkpoints/g1_m1/adapter_model.safetensors | 98,882,136 | b5278b1fed5e784945b6770b9abfd3b8d0cede946d2d98ffd771871f2fc86090 |

远端引用但镜像位置未发现的权重例子（完整列表及来源 JSON pointer 见 JSON）：

- `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_challenger/resume_u0000/adapter_model.safetensors`；证据 `outputs/selfgrok/verge_minimal_20260908/evidence/q_ignition_review_20260908/READY.json`。
- `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_challenger/resume_u0000/state.pt`；证据 `outputs/selfgrok/verge_review_20260908/evidence/checkpoint_state_evidence.json`。
- `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000/adapter_model.safetensors`；证据 `outputs/selfgrok/verge_minimal_20260908/evidence/q_ignition_review_20260908/READY.json`。
- `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000/state.pt`；证据 `outputs/selfgrok/verge_review_20260908/evidence/checkpoint_state_evidence.json`。

## 大文件与文件类型

| 文件类型 | 数量 | Bytes |
|---|---:|---:|
| .json | 1,734 | 127,957,943 |
| .safetensors | 1 | 98,882,136 |
| .jsonl | 43 | 20,728,772 |
| .tar.gz | 36 | 13,862,300 |
| .py | 509 | 3,463,476 |
| .err | 17 | 608,720 |
| .pyc | 19 | 425,806 |
| .md | 42 | 274,000 |
| .log | 90 | 273,769 |
| .sbatch | 170 | 124,901 |
| .ebnf | 1 | 119,322 |
| .mjs | 12 | 56,096 |
| .patch | 2 | 38,664 |
| .out | 17 | 37,524 |
| .txt | 17 | 35,549 |
| .sh | 27 | 22,514 |
| .before_search_route_20260907 | 2 | 15,924 |
| .yaml | 2 | 5,370 |
| .manufactoria | 11 | 4,931 |

≥50 MiB 文件 1 个；≥100 MiB 文件 0 个。这是大小分桶，不替代上传服务限制核验。

## 凭据候选扫描

扫描 5,824 个文本对象（包含 tar 文本成员），候选位置 0 个。仅记录路径、类型和行号，未保存匹配值或片段。
本次规则扫描未发现明显凭据候选；这不等于二进制、嵌套压缩内容或任意自定义密钥格式都已排除。

扫描期间文件变化/读取问题：0；tar 错误：0。本 JSON 保存每个普通文件的大小及 SHA256、目录递归统计、tar 成员清单和远端引用证据。

## 旧 manifest 的相对产物引用补充

旧工程 169 个 manifest 另含 132 个不同相对项目路径；按类型为 `{"checkpoints": 128, "data": 4}`。镜像位置未发现的分类为 `{"checkpoints": 128, "data": 4}`。因此，上面的 4 项绝对权重文件引用不代表旧 checkpoint 总量；另有 128 个不同 checkpoint 相对目录引用未在旧工程镜像位置发现。

这些是文件盘点证据，不确认路径所指训练是否成功、权重版本是否正确或远端当前是否仍存在。完整相对路径和来源 JSON pointer 见 `old_manifest_artifact_references.json`，同时已并入主 inventory。

- `checkpoints/decisive_mistral_v5_5__oracle/resume_u0004`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_mistral_repair_acceptance.json`。
- `checkpoints/verge_book_v1_frozen_r00_challenger/resume_u0000`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_frozen_r00_complete.json`。
- `checkpoints/verge_book_v1_frozen_r01_candidate_1/resume_u0015`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_sampling_incident.json`。
- `checkpoints/verge_book_v1_frozen_r01_direct/resume_u0016`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_sampling_incident.json`。
- `checkpoints/verge_book_v1_initial_challenger/resume_u0000`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_frozen_r00.json`。
- `checkpoints/verge_book_v1_initial_solver/resume_u0000`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_frozen_r00.json`。
- `checkpoints/verge_book_v1_outcome_r00_challenger/resume_u0000`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_outcome_r00_complete.json`。
- `checkpoints/verge_book_v1_uncertainty_r00_challenger/resume_u0001`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_uncertainty_r00_complete.json`。
- `checkpoints/verge_book_v1_verge_r00_candidate_3/resume_u0029`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_verge_r00_complete.json`。
- `checkpoints/verge_book_v1_verge_r00_challenger/resume_u0001`；来源 `outputs/selfgrok/verge_review_20260908/review_copy/manifests/verge_book_v1_verge_r00_complete.json`。
