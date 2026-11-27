# Run 5: Three-Stage Flat-Ground Walking

## Stages

| Stage | Commands | Passing criteria |
|---|---|---|
| 1: Standing | Zero velocity | All of the latest 10 episodes survive |
| 2: Fixed-speed walking and stopping | 80% forward at 0.3 m/s, 20% stop | All 10 episodes survive, with mean velocity error <5% |
| 3: Random-speed walking and stopping | 70% forward speed sampled uniformly from 0.3–1 m/s, 30% stop | Same as Stage 2; save the model and end training when passed |

Use flat ground throughout, with 20-second episodes and zero lateral and turning commands. Each command lasts for a duration sampled uniformly from 3–8 seconds. Consecutive samples may produce the same command.

## Workflow

Both new and existing models are checked from Stage 1, keeping their current parameters. At each stage, first run 10 episodes with the model frozen. If they pass, check the next stage; otherwise, train the current stage. During training, advance as soon as the latest 10 episodes meet the criteria, without an additional evaluation.

Count episodes across all parallel environments, including failures. Episodes ending on the same step are ordered by environment index; with multiple GPUs, order by rank first. Reset all environments and clear statistics when advancing. Save the current model if the iteration limit is reached. Reward and Loss do not determine progression.

## Velocity Error

    e(t) = norm(actual planar velocity - target planar velocity) / max(norm(target planar velocity), 0.3 m/s)

Average over all control steps within each episode, then average the latest 10 episode means. Include starting, slowing down, and lateral errors. For stop commands, use 0.3 m/s as the denominator. The window mean must be <0.05; individual episodes do not each need to meet this threshold. Stage 1 checks survival only.

## File Changes

- **env/mdp/curriculum.py**: Adds `VelocityStageCfg` and replaces `StandingToWalkingCommand` / `StandingToWalkingCommandCfg` with `VelocityCurriculumCommand` / `VelocityCurriculumCommandCfg`. `make_velocity_command_cfg()` defines all settings; `begin_stage()` switches stages, `record_step()` / `reset()` collect episode results, `record_results()` checks acceptance, and `curriculum_episode_complete()` controls episode duration. Reuses the sampler in `command.py`.
- **runner/velocity_runner.py**: `learn()` implements the check/train loop; `_check_stage()` checks the frozen model, and `_collect_results()` aggregates episodes. `save()` / `load()` persist the stage and support older checkpoints. Training rechecks from Stage 1; play uses the saved stage.
- **env/velocity_env.py**: Connects the new termination callback and uses the shared episode duration.
- **cfg/leo_quadruped/env_cfg.py, README.md**: Update the explanation of how play uses the checkpoint stage.

Settings are defined only in `curriculum.py`; `command.py` and PPO hyperparameters are unchanged. Metrics under `Curriculum/velocity/` report the stage, window episode count, survival rate, mean error, and passing status.
