# VERGE 通用 DSL 约束与同分布 loss

这是可接入 HF `generate(logits_processor=...)` 的 Solver 解码器备选，已完成真实已安装环境的 CPU token-mask 验证。它不加载模型权重、不提交 GPU、不修改旧 release，不包含目标参考程序。语义正确性仍由原 binary full-pass verifier 判断；格式合法不等于任务成功。

## 现场依据与环境

2026-09-08 定点读取了以下 selfgrok 文件，未从其他项目推定环境：

| 已安装文件 | 核查事实 |
| --- | --- |
| `envs/vllm/lib/python3.11/site-packages/xgrammar/contrib/hf.py` | HF processor 在首次调用初始化 matcher，其后接受上一 token，再给下一 token 填 mask；每个 `generate` 需要新 processor。 |
| `.../xgrammar/tokenizer_info.py:173` | `from_huggingface` 支持显式 **模型** vocab size 与 stop token IDs；不能仅用 tokenizer 长度。 |
| `.../xgrammar/matcher.py:304,325` | 支持 `accept_string` 接受固定 prompt prefill；生成动作必须逐个 `accept_token`；packed mask 的 1 位代表允许。EOS 只有完整 grammar 后可接受。 |
| `.../xgrammar/compiler.py:310` | 可编译通用 EBNF grammar。 |
| `envs/vllm/.../vllm/v1/structured_output/backend_xgrammar.py:60` | 现场 vLLM 的 HF tokenizer 路径同样使用 `TokenizerInfo.from_huggingface(..., vocab_size=...)`。 |
| `envs/uncertainty/.../transformers/generation/utils.py:1293–1356` | 自定义 processor 后还可能应用 temperature、top-k、top-p、top-h、min-p 等 warper；只重放语法 mask 却遗漏这些变换会造成训练分布不一致。 |
| 原 `src/verge_repair_grammar.py` | 原脚本重放 Challenger JSON schema 的 mask；不是现成 Solver DSL grammar，未复用其 JSON 奖励或旧科研协议。 |
| `vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py:359–484` | 固定唯一 START/start 与 END/end，标准 6 类中间节点，声明及引用校验；执行器另有 1000 步失败上限。 |

版本实测：训练环境 `uncertainty` 的 Transformers **5.16.1**，没有 xgrammar；`vllm` 环境的 Transformers **4.57.6**、xgrammar **0.2.4**。因此没有把 vLLM 整个 site-packages 注入训练进程，也没有安装/覆盖原环境。常驻 CPU worker 在现有 vLLM Python 中编译 grammar，经 stdin/stdout RPC 返回 packed mask，HF 训练进程保留原 Torch/Transformers。

实际 tokenizer 来自 `/blue/du.j/jinjiaguo/self grok/models/Ministral-3-3B-Instruct-2512-BF16`，模型 vocab 为 **131072**，EOS=**2**。CPU 测试确认两个环境的词表到 ID 映射完全一致。

## Grammar 的范围

`dsl_grammar.py` 枚举节点数 2..32（含 start/end）。节点声明规范为 start、n0..nK、end。对每个实际节点数分别生成引用集合，所有 routes 只可指向该程序声明的节点或 NONE，允许向前、向后、自环和回到 start。保留四种 PAINTER 与两种 PULLER，每个 puller 显式给出其两色与 EMPTY 分支。省略的失败分支可表达成显式 NONE；不额外加参考解、目标输出前缀、固定拓扑、固定节点类型序列或成功轨迹。

规范命名和声明顺序保留至 32 节点图的表达能力，但改变了文本采样策略，必须对 base、direct、curriculum、评估统一使用并在本轮配置冻结。它不能保证所有采样都在 token cap 内完成；被 cap 截断的前缀仍应进入完整分母并按 verifier 判失败。严格 verifier 继续拒绝错误输出、NONE、执行错误和超时。

固定格式前缀仍为 ` ```manufactoria\nSTART start:\n    NEXT `（首个反引号前无空格），只进入 prompt。grammar worker 已消费该字符串；后续 mask 从 **原生 suffix** 的第一个 token 开始。完整 grammar 的闭合围栏后只允许 EOS；生成 EOS 纳入预算/loss，EOS 后 batch pad 排除。

## 接口与训练约束

```python
from hf_grammar_bridge import (
    GrammarService, HFGrammarLogitsProcessor,
    native_sampling_kwargs, masked_suffix_loss,
)

with GrammarService(vllm_python, actual_model_directory) as grammar:
    # input_ids 已含统一 PROGRAM_PREFIX；每次 generate 都创建新 processor。
    generated = model.generate(
        **batch, logits_processor=[HFGrammarLogitsProcessor(grammar)],
        **native_sampling_kwargs(), max_new_tokens=2048,
        eos_token_id=grammar.metadata['eos_token_ids'], pad_token_id=tokenizer.pad_token_id,
    )
    # completions 是实际 native suffix IDs，含生成 EOS、去除其后 pad。
    # binary_advantages 仅由当前 prompt 的实际 binary full-pass reward 得到。
    # 全组 reward 相同：整个 optimizer/scheduler step 精确跳过。
    if any(binary_advantages):
        optimizer.zero_grad(set_to_none=True)
        generated_tokens = sum(len(ids) for ids in completions)
        for ids, advantage in zip(completions, binary_advantages):
            packed, grammar_terminated = grammar.replay(ids)
            loss, _ = masked_suffix_loss(model, prompt_ids, ids, advantage, packed)
            (loss / generated_tokens).backward()
        # 使用现有非有限梯度检查、clip、optimizer.step、scheduler.step 与日志。
```

**不能**将 guided generation 接回旧 unmasked `train_step`。这里每个动作的 log-prob 为 `logit[action] - logsumexp(logits[allowed])`；mask 与生成端逐位一致，prefix 不进入 loss。不使用 partial reward 或参考模型/KD。Caller 继续负责 fresh optimizer、dropout=0、beta=0、weight decay=0，以及常数组精确跳过。

`native_sampling_kwargs()` 明确 temperature=1、top-p=1、top-k=0（HF 中关闭 top-k，对应原 native 的 -1），禁用其他概率 warper/偏置。不要额外加入未在 loss 中重建的 processor、warper、beam、speculative decoding 或重排 batch。实例固定顺序，processor 验证每调用恰好前进一步。暂停/到达 cap 的样本可重放未闭合 grammar；这不把失败样本排除。

## 已完成 CPU 验证与成本

- `cpu_xgrammar_tests.log`：真实 vLLM 环境 **9 项测试通过**。包括 2/32 节点、全类型、自环、NONE/start/end routes、未声明引用/重复 ID/错误类型拒绝、提前 EOS 拒绝；验证 masked logp 与 unmasked logp 不同，并验证禁止动作梯度为零。
- `cpu_actual_tokenizer_bridge.json`：真实训练环境 → CPU worker，51 个 native suffix tokens（含 EOS）；生成/重放 mask 逐位相同，masked logp 与采样分布一致，autograd 有限。单行逐步 mask RPC 0.154 秒，replay 0.0066 秒。
- `cpu_actual_batch8.json`：**8 行不同 EOS 时刻**，native tokens 为 `[7,21,35,48,61,91,121,629]`，合计 1013，最长 629 步。全部生成 mask 与训练重放相同；EOS 后 pad 不进入重放；真实 vendor parser 验证全部 8 个 fixture 合法、全部引用存在。其间没有任何模型生成或 target 成功率统计。
- Batch 8 worker 完整冷启动（含库导入）19.53 秒，其中 tokenizer/grammar 阶段 1.81 秒；629 步 CPU mask/RPC 检查 8.67 秒，8 条 replay 合计 0.145 秒。1013 tokens 的 packed masks 为 16,596,992 bytes，约 **16 KiB/token**。估算 2048 tokens 的单条完整 mask 约 32 MiB；逐条重放/反传可限制暂存。实际 GPU 运行的额外墙钟仍需实测，不能把 CPU fixture 时间当 GPU 吞吐。

此结果证明接口可在现有环境连接，不证明 constrained policy 对目标更有效。还没有为本模块运行 GPU；root 接线后需保留独立运行日志。

## Challenger 的通用 JSON schema 模式

后续增加的 JSON 模式复用同一个 CPU worker/RPC，不改 DSL 默认行为：

```python
with GrammarService(vllm_python, actual_model_directory,
                    json_schema_path=actual_run_directory / 'challenger_schema.json',
                    prefix='', log_name='challenger_grammar.stderr.log') as grammar:
    # 普通 chat prompt，无 DSL 格式前缀；使用初始化的独立 Challenger adapter。
    ids = challenger.generate(
        **chat_batch, logits_processor=[HFGrammarLogitsProcessor(grammar)],
        **native_sampling_kwargs(), max_new_tokens=proposal_cap,
        eos_token_id=grammar.metadata['eos_token_ids'], pad_token_id=tokenizer.pad_token_id,
    )
```

Caller 提供并冻结完整 JSON schema，可以放在任意已授权 selfgrok run 目录；服务只读此文件。支持显式 consume `prefix=''` 或 `'{'`，prefix 若非空必须已经进入 prompt，最终 JSON 还原时只补回这段 **固定格式**；不能填补截断 proposal 的未知字段。schema 控制允许的 `base_checkpoint`、三个 curricula 各三 stage，以及 catalogue 的 stage_id/kind/spec 绑定；模型实际选择 proposal，而不是由本库写定课程。

`json_schema_support.py` 对对象和数组 `const` 进行**显式等价展开**：固定对象的每个字段递归 const、required 全部字段、禁止额外字段；固定数组使用 prefixItems、固定长度、禁止额外项。Worker metadata 记录 `structured_const_expansions` 和展开后的 schema SHA。遇到不支持的冲突约束或编译失败直接报错，不静默放松 schema。原 schema 不被改写。

新增 JSON CPU 测试后，实际 vLLM 环境 **14 项 DSL+JSON 测试全部通过**，见 `cpu_dsl_and_json_tests.log`。JSON 测试覆盖固定 base、kind/spec 对应、三条课程各三 stage、额外字段拒绝，以及空/开花括号 prefill。`cpu_actual_json_bridge.json` 使用真实 catalogue 9 个选项与真实 tokenizer，仅在内存构造明确标注的 CPU fixture 检查 masks，**没有写入或提交手工 proposal**、没有加载模型。27 处 structured const 展开后均成功编译，2064 个 native tokens（含 EOS）全部接受；冷启动 14.04 秒，replay 1.36 秒。

最短 identity catalogue 规格重复成 9 个 stage 的 CPU fixture 已有 **4215 字符、2064 tokens**。因此完整 spec 的 Q cap=2048 不够；正式 Q cap 应由 caller 在其总预算内明确设置，并保留失败/截断结果。此长度是合成接口检查，不是模型 proposal 或实验结论。本次模式只用于真实 Challenger 采样；没有新增在线 Challenger GRPO，也没有自动触发 RFT。

实际远端目录为 `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/decoding`。本目录代码、测试、结果与 CPU worker 缓存均留在 decoding 内。

复现入口：在该目录，载入 `python/3.11`，显式 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`，把 `TVM_FFI_CACHE_DIR`、`XDG_CACHE_HOME`、`MPLCONFIGDIR`、`TMPDIR` 设为本目录子目录，然后：

```bash
"/blue/du.j/jinjiaguo/self grok/envs/vllm/bin/python" -B -m unittest -v test_decoding_cpu
"/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python" -B cpu_batch_probe.py \
  --worker-python "/blue/du.j/jinjiaguo/self grok/envs/vllm/bin/python" \
  --model "/blue/du.j/jinjiaguo/self grok/models/Ministral-3-3B-Instruct-2512-BF16" \
  --parser "/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py" \
  --output cpu_actual_batch8_NEW.json
```
