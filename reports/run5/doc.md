# Run 5：三阶段平地走路

## 阶段

| 阶段 | 指令 | 通过条件 |
|---|---|---|
| 1：站立 | 零速度 | 最近 10 回合全部存活 |
| 2：固定速度走停 | 80% 以 0.3 m/s 前进，20% 停止 | 全部存活，且 10 回合平均速度误差 <5% |
| 3：随机速度走停 | 70% 从 0.3–1 m/s 均匀抽取，30% 停止 | 同第二阶段；通过后保存并结束训练 |

全程平地，每回合 20 秒，侧向和转向指令为零。每条指令保持 3–8 秒，均匀随机，允许连续抽到相同指令。

## 流程

新模型或已有模型都从第一阶段检查，保留当前参数。每阶段先固定模型跑 10 回合；通过就检查下一阶段，未通过就训练。训练中最近 10 回合达标便晋级，不额外验收。

10 回合按所有并行环境合计，失败回合也计入。同一步完成的回合按环境编号排序，多 GPU 时先按 rank 排序。晋级时重置所有环境并清空统计；达到最大迭代数则保存当前模型。Reward、Loss 不参与晋级。

## 误差

    e(t) = norm(实际平面速度 - 目标平面速度) / max(norm(目标平面速度), 0.3 m/s)

先对每回合全部控制步求平均，再平均最近 10 回合。包括起步、减速和侧向误差；停止时用 0.3 m/s 作分母。要求窗口平均值 <0.05，不要求每回合单独达标。第一阶段只检查生存。

## 文件修改

- **env/mdp/curriculum.py**：新增 VelocityStageCfg；用 VelocityCurriculumCommand / Cfg 替换原来的 StandingToWalkingCommand / Cfg。make_velocity_command_cfg() 集中定义全部参数；begin_stage() 切换阶段，record_step() / reset() 统计回合，record_results() 判断通过，curriculum_episode_complete() 控制回合时长。继续复用 command.py 的采样器。
- **runner/velocity_runner.py**：learn() 实现检查与训练循环；_check_stage() 固定模型检查，_collect_results() 汇总回合。save() / load() 保存阶段并兼容旧 checkpoint。训练启动时重新检查，play 使用保存的阶段。
- **env/velocity_env.py**：接入新终止回调，引用统一的回合时长。
- **cfg/leo_quadruped/env_cfg.py、README.md**：更新 play 使用 checkpoint 阶段的说明。

参数只在 curriculum.py 定义；command.py 和 PPO 超参数未修改。日志使用 Curriculum/velocity/ 下的阶段、窗口回合数、生存率、平均误差和通过状态。
