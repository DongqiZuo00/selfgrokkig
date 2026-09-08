# VERGE 下一步与本轮交付
2026-09-08。

**本轮完成：** 原始两文全文阅读、当前工程/指令/数据/模型元数据核查、24轮96端点的保存计数复算、64条旧 completion 的 binary verifier 重放，以及独立副本的3处最小修复。工程回归68项通过；另10项合成规格测试通过。生产源文件 SHA256 与审计快照一致，原有4个跟踪文件改动保留。

交付位于：
- 远端：`/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_review_20260908`
- 本地：`C:/Users/ddong/Documents/Codex/2026-09-06/hipergator-x20/outputs/selfgrok/verge_review_20260908`

METHOD_REVIEW.md 给独立科研判断；IMPLEMENTATION_AUDIT.md 对照真实组件。review_copy 保存修复后的源码/配置，patches/minimal_fixes.patch 是明确差异，evidence 保留失败和通过日志、原文哈希与现场证据。

## 尚未运行
没有提交 Gate 0–3、没有重新筛选 target、没有加载完整模型作 GPU smoke、没有取消已有作业、没有修改原训练配置或 checkpoint、没有恢复 CAFD/T1/T2/KD/RL routing/MPC。
已有 target-only search 正在使用旧协议；其继续推进不由本轮触发，也不能记作新版闭环完成。

## 已确认的阻塞范围
1. **真实 test→class 映射缺失。** 阻塞正式 f、frontier、以其为依据的课程选择/分格/新结果复算。生成器中 basic/corner 等步骤不是类别。不能合成单一类来绕过。
2. **逐阶段 Delta/telescoping 缺失。** 阻塞原文规定的 per-stage context/经验记录和完整课程分解；没有用零或删除字段代替。完整课程 Gamma/kappa 的数学定义已给出，可以继续独立测试。
3. 新 archive/hint-validator/J+ RFT 尚未实现；不能仅改 KL 和条件维数就启动闭环。
4. 正式运行前需确定 target/stratum、backbone/tokenizer revision/初始 checkpoint 的新版采用依据；B、alpha、eta、q、N_min、T、CI/seed 分组，以及两卡还是四卡的唯一资源合同。旧协议数字只是实现事实。
5. 无 parse-valid、全部条件饱和、完全并列、无 hint stage 的 validator 适用方式和参数元组无泄漏口径仍要明确。METHOD_REVIEW 中列出的主张/机制修改是建议，未经批准。

用户已确认前两项没有可补充原始依据，本轮不再重复索要。它们不阻塞 CPU 数据完整性、binary reward 和 optimizer 测试，但阻塞依赖它们的正式闭环。

## 下一条可复制命令：只复跑独立 CPU 验收
从本机 PowerShell：
```powershell
wsl.exe ssh HiPerGator 'bash "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_review_20260908/RUN_CPU_TESTS.sh"'
```
脚本使用现有 uncertainty Python、单线程 CPU、CUDA_VISIBLE_DEVICES 为空；无 sbatch、无模型采样。日志用时间戳生成，不覆盖旧记录。依赖路径及同组命令已在本轮验证；最终脚本运行结果保存于 evidence/retest_*.log 和 spec_*.log。需求为0 GPU，实测轻量 CPU 测试，不占用192GB/2–4 B200训练额度。

正式下一步应先形成有来源的 class 映射与 Delta 定义，单独审阅方法主张，再做新版版本化实现和加载 smoke 的具体配置/预算。本轮没有提供会自动启动正式训练的命令。

