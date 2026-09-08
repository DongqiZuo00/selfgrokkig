# 最小实验已完成；下一研究目标

三轮探索已结束，第三轮完成了含真实Solver更新的一轮VERGE开发实验：三条课程与direct各65,536生成训练token，g1发生1次binary mixed更新，全部阶段探针、target-only尾段、最终P32评价、评分和独立复算通过。三条Gamma均为0，目标full pass均未出现，J+为空；未观察到目标迁移收益。

累计1,039,681生成token、2.5425 B200小时；指定新增作业峰值2 B200/128GiB。87项当前执行路径CPU测试通过，另有独立vendor重放和成本审计工具测试。所有新增作业已退出，旧成果和其他作业保留。

## 可复核交付

本地目录：`C:/Users/ddong/Documents/Codex/2026-09-06/hipergator-x20/outputs/selfgrok/verge_minimal_20260908`。

远端镜像：`/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908`。

- `EXPERIMENT_RESULT.json`：最后验收PASS，明确最小流程完成、target未解决、有效性未成立。
- `runs/round3_result/`：真实exchange、Gamma/CI、Delta、archive、事件、lineage及J+/RFT状态。
- `runs/round3_g1_41420288_0/execution/checkpoints/g1_m1/`：真实更新LoRA，本地/远端均完整保留；SHA为b5278b1fed5e784945b6770b9abfd3b8d0cede946d2d98ffd771871f2fc86090。
- `evidence/ROUND3_UPDATE_REPLAY.md`：原vendor重放及全部LoRA张量直接比较。
- `evidence/COST_THREE_ROUNDS.md/json`、`FINAL_ACCEPTANCE.log`、三份`round*_independent_score_audit.json`：真实预算、缓存去重与最终检查。
- `evidence/COMMANDS.md`、各runtime脚本、configs和原始runs日志：实际修改与复现命令。

一条已执行验证、可重复使用的只读命令，在上述远端目录执行：

```bash
python3 runtime/round3_results.py --compact
```

此命令无GPU工作。它应显示四分支complete、每支65,536训练token、g1一次更新及256条新增评价。最终评分产物已存在，不需重复提交训练或覆盖结果。

## 仍未完成的研究工作

下一目标是找到能产生无hint目标收益的课程。此次点燃来自identity，目标相邻的变换stage仍没有稳定mixed证据；下一次预算应优先用于这些stage的可点燃性探针，再决定新的匹配训练。三类stage来源继续保留；若采用hinted target，先按原四项validator完成32×8两臂、共512条的hint-regret探针，hint来源只能是可追溯的已验证程序库。

正式P1–P7、Gate0–3、多seed比较、开放变换提案、hint迁移、持续多轮多incumbent archive与跨seed正例RFT均未完成。当前scorer每轮从base重建archive/buffer、Q RFT仅输出计划，持久闭环的生产接线仍须实现和测试；不能将本轮验收解释为这些部分已完成。完整方法研究范围没有删除。

本次最小实验没有未决输入阻塞。原文缺少的测试类映射和per-stage Delta来源仍未恢复；采用的是用户放宽后明确版本化的新操作定义，不再冒充原稿定义。g1首阶段有公开的人为点燃先验，也不构成无约束Challenger发现课程的效果证据。
