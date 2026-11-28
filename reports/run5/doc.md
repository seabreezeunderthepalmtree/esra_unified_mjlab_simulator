# Run 5: Five-Stage Flat-Ground Walking

## Stages

| Stage | Commands | Progression |
|---|---|---|
| 1: Standing | Zero velocity | Advance when training is stable |
| 2: Fixed-speed walking and stopping | 80% forward at 0.3 m/s, 20% stop | Same stability rule |
| 3: Random-speed forward walking | 70% forward speed sampled uniformly from 0.3–1 m/s, 30% stop | Same stability rule |
| 4: Add backward and lateral walking | Forward at 0.3–1 m/s; backward, left, or right at 0.3 m/s; 30% stop | Same stability rule |
| 5: Expand all four directions | Forward, backward, left, or right at 0.3–1 m/s; 30% stop | Save and finish when stable |

Keep flat ground, 20-second episodes, zero turning commands, and the existing disturbances. Command durations are sampled uniformly from 3–8 seconds; consecutive commands may be identical. Each environment samples independently.

In Stages 4–5, first choose stop with 30% probability; otherwise choose one of the four body-frame directions uniformly (25% each among moving commands). Each moving command activates exactly one planar axis, so there are no diagonal commands. Variable speed magnitudes are sampled uniformly within their ranges. Stages 1–3 keep their existing command behavior.

## Stability Rule

After every training iteration, compare the latest 50 iterations with the preceding 50 from the same stage. Both mean reward and mean value loss must change by less than 5%:

    change(X) = abs(mean_recent_50(X) - mean_previous_50(X))
                / max(abs(mean_previous_50(X)), 1e-8)

Reward is the mean total environment reward over all rollout steps and parallel environments in one iteration. Value loss is the PPO update result, averaged across GPUs when applicable. Policy/surrogate loss is logged but does not gate progression. Non-finite windows cannot qualify.

There are no survival or velocity-error gates and no frozen-model checks. Each visited stage trains for at least 100 iterations. When stable, save a stage checkpoint, append an alert, reset all environments and stability history, and immediately train the next stage. Stage 5 stability ends training. Reaching the iteration limit saves the current model.

## Starting Stage

New and resumed runs default to Stage 1. Add `--agent.start-stage 4` with the existing resume options to continue a trained model with the new directions. Valid starting stages are `1`–`5`. The selected stage starts with fresh stability history; loaded model and optimizer parameters are retained. Later stages advance automatically. Play uses the stage saved in the checkpoint.

## Alerts

The repository-root file `alert` receives one English message per stage transition, including the stage, iteration, window means/changes, and full checkpoint path. Snapshots are saved as `stage_<stage>_stable_model_<iteration>.pt` in the new run directory. Training continues without waiting for review. The alert file is local to the training machine; it does not automatically message this chat.

## Interactive Inspection

Run `uv run --no-sync python src/esra_unified_mjlab_simulator/scripts/leo_policy_viewer.py --checkpoint-file "/path/to/model.pt"`. The browser provides random/manual commands, a random-push toggle, and forward/backward/left/right kick buttons. Random commands automatically use the training profile associated with the checkpoint. It reuses training noise, push settings, and episode resets in one environment without training. See the README for controls.

## File Changes

- **env/mdp/curriculum.py**: `VelocityStageCfg.additional_directions` selects optional backward/lateral speed ranges. `make_velocity_command_cfg()` centrally defines all five profiles and the common stability rule; `_apply_stage_cfg()` supplies the selected profile to the command sampler.
- **env/mdp/command.py**: Adds `UniformVelocityCommandCfg.AdditionalDirections` and optional four-direction sampling in `_resample_command()`. Forward speed uses the existing x range; backward and lateral ranges specify positive speed magnitudes. Uses strict probability comparisons so zero-probability modes stay disabled. Stopping and command timing retain their existing behavior.
- **cfg/leo_quadruped/algo_cfg.py**: Adds `VelocityOnPolicyRunnerCfg.start_stage`, exposed as `--agent.start-stage` with default `1`.
- **runner/velocity_runner.py**: Starts training at the selected stage and advances on stability. Saves stage snapshots and appends their paths to `alert`. Retains the inference-mode reset fix and old checkpoint compatibility.
- **alert**: New root-level notification file.
- **README.md**: Documents the starting-stage option, progression, and checkpoint/alert instructions.
- **scripts/leo_policy_viewer.py**: Adds `ViewerVelocityCommand` / `ViewerVelocityCommandCfg` for browser command control, `PushControls` / `install_push_controls()` for gated training pushes and queued manual kicks, and `LeoPolicyViewer` for playback and readouts. Training settings remain in their existing configuration files.

Reward functions, disturbances, and PPO hyperparameters are unchanged. The existing runner handles all five stages by reading the stage count. `Curriculum/velocity/` logs the stage, stage iteration count, window means, relative changes, stability, and completion. No commands, tests, or training were run; validation was limited to static inspection.
