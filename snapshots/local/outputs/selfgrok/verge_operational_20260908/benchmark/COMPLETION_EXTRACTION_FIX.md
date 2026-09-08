# 原始模型输出接口修复：CPU 重放 job 41369486

实际 smoke 暴露了一个实现差异：新 `evaluate_program` 将包含解释和 Markdown fence 的整条 completion 直接交给 DSL parser；原工程 `common.py` 一直先调用 `extract_program`。这会把完整、合法但带 prose/fence 的程序误记为 parse failure。

已在独立版本的 `benchmark.py` 原样复制 `review_copy/src/common.py::extract_program` 并在 parse 前调用。测试比较两个函数的 AST，确认正文语义相同。保留原 fenced 匹配及 START/END fallback；没有按 parse/reward 选择其他代码块，没有补写模型未生成的 prefix、END 或代码。输入 raw string 未改动，原 smoke 原始记录也未重写。

`revisions/pre_smoke_41369486.py` 保留运行时原 verifier，SHA256 `3356f8efe2240fe938ffe48711ab48d1e0c5be3571d216183584062a359b6842`，与 job `preflight` 中的实际源代码 SHA 一致。修复后 SHA 为 `cb59f723f4ef5148ec0de308b7d9e15d8b2093b798e37d8f90a4a1271ef751d9`。

原始 32 条 saved sampled_group 在 CPU 上逐条通过“原 raw verifier / 修复后 extraction verifier”重放，使用当时相同 train row 文件和原 parser。所有旧 parse、binary reward、f、class rates、逐测试 pass 与保存记录完全重现。重放前后原 `events.jsonl` 和 `summary.json` 的 SHA 均相同，原日志的 19 项 hash chain 验证通过。

| 项目 | 原 raw 入口 | extraction 修复后 |
|---|---:|---:|
| parse-valid | 0/32 | 1/32 |
| binary full pass | 0/32 | 0/32 |
| exact-pass tests | 0/1152 | 0/1152 |

17 条 completion 的提取文本发生变化。全部 32 条原生成均达原 512-token cap；原生成总量仍为 16,384 tokens。修复没有新的模型生成、GPU 使用或 optimizer step。这些输出还受到该 smoke 的 prompt/prefill/长度配置影响，不能据此认定新 benchmark 的目标 regime。

新增 5 项测试并连同既有测试在远端原环境执行，**15/15 通过**：原函数 AST 一致；完整 fenced+prose reference；缺 END 的截断保持失败；未闭 fence 但存在完整 START/END 按原 fallback 可提取；首个 fenced 完整但非法程序不会因后面还有有效 reference 而被替换。原 parser 下也实际执行这些正负例。原函数优先搜索匹配的 fenced block，然后才使用 START/END fallback，这个既有优先级原样保留。

证据：

- `extraction_cpu_tests.log`
- `../replay_gpu_smoke_41369486.log`
- `../runs/gpu_smoke_41369486_replay_20260908T065838_098026/replay_summary.json`
- 同目录 `replay_records.json`：逐条 raw、extracted text、原事件位置/hash、前后 verifier结果。

可复制 CPU 命令（self grok 既有环境，CUDA_VISIBLE_DEVICES 为空）：

```bash
python benchmark/test_benchmark.py --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py'
python runtime/replay_gpu_smoke.py --output runs/<fresh-replay-directory>
```

此修复只处理 completion extraction；原生产入口中的预填前缀和采样上限差异由主任务单独记录、验证，不能在本次 CPU 重放里替模型补写。
