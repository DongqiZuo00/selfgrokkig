# 执行状态与下一步

本轮已经把两项缺失定义变成明确的新版本，实现并测试协议、三种 stage、archive、Challenger 正例 buffer、generated-token 预算和真实 Torch Solver 更新。采用同一个字符串变换的 RB 输入候选 benchmark，原始四色 benchmark、代码、checkpoint 与正在运行的作业保留。

## 本轮 GPU 验证与修复

`41369486`：`verge-op-smoke`，1 B200、64G、20 分钟硬上限，最多 4×8×512=16,384 生成 tokens。加载实际初始 Ministral Solver LoRA，只读 train split，无 hint；不读取 heldout/test 模型结果。随机采样若形成 mixed binary reward group 才更新；全奖相同则验证精确跳过。输出独立新 adapter，不覆盖初始 checkpoint。

该作业已完成，Slurm 实际 129 秒。32 条输出的 full pass 为 0，所有同奖组精确跳过；发现输出提取和代码格式前缀的接口遗漏，已实际修复。原输出 CPU 重放 parse 从 0/32 变成 1/32，full pass 仍为 0。原来的 summary 与原输出保留。

`41370126` 是格式修复补测：1 题×8 条、cap 2048，仍从同一个初始 Solver 加载，恢复原工程的通用代码前缀。申请 1 B200、64G、17 分钟；连同第一轮 129 秒，总 GPU 硬上限仍低于 20 分钟。参见独立 `RUN_GPU_SMOKE_PREFILL.sbatch`、`SUBMIT_GPU_SMOKE_PREFILL.sh` 和 `runs/gpu_smoke_prefill_*`。

这两次不是整轮 VERGE、不是 regime 结论，也不是 Gate 0–3 套件。最终状态和实际费用见 `FINAL_VALIDATION.json` 与 sacct 日志；排队时间不计训练时间。未通过的程序保留，GPU 上未出现混奖更新时须明确记录，不能用 CPU TinyLM 更新测试代替真实 backbone 的更新结果。

最终状态：两作业均完成且 exit 0。补测 12,483 tokens、parse 0/8、full pass 0/8，未发生 optimizer step；两次合计 242 秒，即 0.067222 B200-h。当前没有本轮仍在排队或运行的 GPU 作业。

再次提交脚本会因已存在的 submission 记录拒绝重复；需要复跑时使用新的独立提交记录，先检查旧作业。原有 VERGE search 作业不取消、不接管。

## 可以直接重复的 CPU 命令

```bash
cd "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_operational_20260908"
bash RUN_CPU_TESTS.sh
```

命令逐项运行协议、stage、真实 Torch、benchmark 与合成集成轮。全部日志写新目录，错误通过 pipefail 返回非零。

## 实施顺序

1. 使用本次已修复的输出提取与格式前缀。接下来优先测试简单注册 stage 的 parse-valid、binary 成功、mixed-group 和实际更新，解决已观察到的非法程序生成问题；随后才检验无 hint target 转移。原始失败、CPU 重放与修复补测分别保留，不使用漏掉前缀的初版 HF 入口推断 benchmark 难度。
2. 用新 benchmark 做小规模 Gate 0/1：binary direct、dense direct、原文保留的可点燃性 oracle/hinted stage 对照。reference 程序只验证可解性，不取代这些学习实验。
3. 将本目录明确接口接入实际 generation/stage dataset/guided proposer/RFT backend，验证一轮共享 base、四分支、三 stage 加 target-only tail、64×8 probe 与64×32 final。使用 `operational_config.json` 的明确开发默认值，并把所有 probe/hint/Challenger token 与 GPU 成本另记。
4. 通过单轮后推进 Gate 2、Gate 3 与 P1–P7 原有比较。保留失败候选与预算，用事先定义的判据决定继续、换候选 benchmark 或得出反证；不缩小研究问题来包装完成。

原文件未固定的 B/alpha/eta/q/N_min/T 已在配置中明确，属于本次开发默认值。正式全套实验的总费用需单独核对，不自动承诺或提交原文约 1425 B200-h 的全部计算。资源仍受用户 4 B200、192G 上限约束。
