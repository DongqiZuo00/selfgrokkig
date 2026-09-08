# 冻结真实一轮的分支作业入口

`run_branch_job.py` 只执行一个已READY的 branch，不提交或取消Slurm作业。与 `RUN_PILOT.sbatch` 仅传 `--output` 的接口相容：必须通过 `VERGE_PLAN_ROOT` 指定实际准备目录、通过 `VERGE_BRANCH` 指定 g1/g2/g3/direct；也可显式传 `--plan-root`、`--branch`。不会搜索“最新”目录推定实验。

本轮准备目录已明确为 `runs/round_prepare_41413994`。截至本次接线，该作业已启动并加载prepare入口，因此它完成后写出的READY仍可能没有后来增加的输入hash字段；wrapper兼容这份真实运行代码，同时会逐项校验 `configs/round1_runtime_contract.json` 中冻结的10个backend、executor、decoder、scorer及输入文件SHA256，并把完整contract复制到每个分支输出目录。

`currentrun_source_bundle = work/selfgrok/verge_minimal_round_bundle.tar.gz`。本次读取该既有准备作业bundle的SHA256为 `990db3526b32bf725de8dd92d14434793bf72a84637ebd3eda0518a53c5fb5be`，大小500392字节。这里登记的是已启动prepare作业的来源；新增wrapper自身随后上传，其源码hash和实际executor hash会单独登记，不能把后来文件说成已加载到正在运行的作业中。

主后端grammar-masked真实mixed梯度更新与checkpoint证明已由作业 `41412533` 实际完成；这项先行证明不是本轮四分支的训练结果。wrapper仍要求每个实际branch自己的run_phase native-token journal、真实mixed update记录及checkpoint文件，才能合并计分。

启动前必须满足：READY状态为 `real_base_and_challenger_complete`；plan和base fragment属于同一明确prepare目录；plan内容hash、真实Challenger raw JSON、base采样合同及base权重都匹配；冻结target view和catalogue读取成功；runtime contract每一个文件hash匹配。不同版本或文件修改会在加载模型前失败，需要明确的新reviewed contract版本。

资源合同为每作业1张B200、64GiB、1小时，新增作业最大并发2，受用户总上限4张B200/192GiB约束；调度端负责维持并发。每branch预算65536 generated tokens，四分支总计262144，probe/Challenger生成另列。wrapper本身不会绕过调度或启动额外作业。

输出目录必须是本发布 `runs/` 下的新目录。`LAUNCH.json` 先保存READY、plan、Challenger、source contract、target/catalogue及实际执行器hash；训练证据写入 `execution/`，成功后顶层产生 `fragment.json`、`COMPLETE.json`，可直接传给execute_round merge。失败会留下 `FAILED.json` 及所有已有部分记录，不覆盖旧实验。

CPU验证：`python -B -m unittest discover -s runtime -p test_run_branch_job.py -v`。5项测试使用明确fixture和注入的空执行器，检查READY、Q来源、环境参数、失败保留、版本hash及contract复制；不加载模型，也不声称完成了真实分支训练。日志为 `run_branch_job_cpu_tests.log`。
