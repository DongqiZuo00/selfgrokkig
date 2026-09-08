# 第二轮部署记录

准备作业 41417194 完成了当前 naming_v2 view 的独立基线与真实 Challenger JSON。完整 plan SHA 为 `6bda7e01b4fbeed9a737bfe979a4903baecdee60d8c554be88f17f4006f86113`。

第一次 CPU preflight 拒绝了启动器哈希：远端 `run_branch_job.py` 为 `85d470bb06740eb18df8fbe447207074e49f4eeaf2042c2fc1d6bd33fc2f8837`，合同要求本地已冻结版本 `cbc0da7784049d2eceff75b14ef2bbc395661debed40af16d64da406959917e9`。逐行 diff 唯一变化是 LAUNCH 记录增加 `branch_wrapper_sha256`；训练逻辑、任务、预算和采样未变。旧远端版本保存为 `evidence/run_branch_job_before_round2.py`；本地保存为 `evidence/run_branch_job_remote_before_round2.py`。随后同步已审核版本，未降低或绕过校验。

第二次 preflight 通过实际 READY、Q 原始输出、base sampling contract、16 项源文件哈希、全部 stage spec 和 target 测试核对，证据为 `runs/round2_cpu_preflight.json`。之后提交数组 41418169，四分支每项 1 B200、64G、最长一小时，同时最多两项；每分支 B=65,536 实际生成 token。

Q 的真实选择为 g1=(prepend_RB,target_small3,prepend_R)，g2=(replace_RB_to_YG,target_small3,target_small2)，g3=(target_small3,prepend_R,prepend_RB)。本轮没有替换其选择，也没有把 mixed-group 探针增加为第五项 validator。

实际提交顺序：

```bash
bash SUBMIT_PILOT.sh runtime/prepare_named_round.py round_prepare_named
module load python/3.11
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 '/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/preflight_round2.py
bash SUBMIT_ROUND2.sh
```

远端上述命令工作目录均为 `/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908`。自有输出路径不会写入其他项目。SSH 登录启动文件偶尔输出其他编辑器 settings 的临时文件错误，这不是本轮运行脚本的操作；没有读取、改写或修复那些编辑器文件。
