# Run 5: Three-Stage Flat-Ground Walking

## Stages

| Stage | Commands | Progression |
|---|---|---|
| 1: Standing | Zero velocity | Advance when training is stable |
| 2: Fixed-speed walking and stopping | 80% forward at 0.3 m/s, 20% stop | Same stability rule |
| 3: Random-speed walking and stopping | 70% forward speed sampled uniformly from 0.3–1 m/s, 30% stop | Save and finish when stable |

Keep flat ground, 20-second episodes, zero lateral/turning commands, and the existing disturbances. Command durations are sampled uniformly from 3–8 seconds; consecutive commands may be identical.

## Stability Rule

After every training iteration, compare the latest 50 iterations with the preceding 50 from the same stage. Both mean reward and mean value loss must change by less than 5%:

    change(X) = abs(mean_recent_50(X) - mean_previous_50(X))
                / max(abs(mean_previous_50(X)), 1e-8)

Reward is the mean total environment reward over all rollout steps and parallel environments in one iteration. Value loss is the PPO update result, averaged across GPUs when applicable. Policy/surrogate loss is logged but does not gate progression. Non-finite windows cannot qualify.

There are no survival or velocity-error gates and no frozen-model checks. Each stage trains for at least 100 iterations. When stable, save a stage checkpoint, append an alert, reset all environments and stability history, and immediately train the next stage. Stage 3 stability ends training. Reaching the iteration limit saves the current model. Stability is a scheduling rule, not proof of motion quality.

New and resumed runs start training at Stage 1 with fresh stability history; loaded model and optimizer parameters are retained. Play uses the stage saved in the checkpoint.

## Alerts

The repository-root file `alert` receives one English message per stage transition, including the stage, iteration, window means/changes, and full checkpoint path. Snapshots are saved as `stage_<stage>_stable_model_<iteration>.pt` in the new run directory. Training continues without waiting for review. The alert file is local to the training machine; it does not automatically message this chat.

## File Changes

- **env/mdp/curriculum.py**: Adds `StageStability`; removes episode acceptance and velocity-error gates. `make_velocity_command_cfg()` centrally defines the three command profiles, 50-iteration windows, 5% threshold, and numerical floor. The existing sampler and episode timeout are retained.
- **runner/velocity_runner.py**: Replaces check/train branching with continuous PPO training and stability-based transitions. Saves stage snapshots and appends their paths to `alert`. Retains the inference-mode reset fix and old checkpoint compatibility.
- **alert**: New root-level notification file.
- **README.md**: Updates progression and checkpoint/alert instructions.

Reward functions, disturbances, `command.py`, and PPO hyperparameters are unchanged. `Curriculum/velocity/` logs the stage, stage iteration count, window means, relative changes, stability, and completion. No commands, tests, or training were run; validation was limited to static inspection.
