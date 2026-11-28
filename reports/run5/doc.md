# Run 5: Seven-Stage Flat-Ground Walking

## Stages

| Stage | Commands | Progression |
|---|---|---|
| 1: Standing | Zero velocity | Advance when training is stable |
| 2: Fixed-speed walking and stopping | 80% forward at 0.3 m/s, 20% stop | Same stability rule |
| 3: Random-speed forward walking | 70% forward speed sampled uniformly from 0.3–1 m/s, 30% stop | Same stability rule |
| 4: Add backward and lateral walking | Forward at 0.3–1 m/s; backward, left, or right at 0.3 m/s; 30% stop | Same stability rule |
| 5: Expand all four directions | Forward, backward, left, or right at 0.3–1 m/s; 30% stop | Same stability rule |
| 6: Add fixed-rate yaw | Planar speeds remain 0.3–1 m/s; independently choose yaw of −0.5 or +0.5 rad/s; 30% full stop | Same stability rule |
| 7: Expand yaw range | Same planar commands; independently sample yaw uniformly from −0.7 to +0.7 rad/s; 30% full stop | Save and finish when stable |

Keep flat ground, 20-second episodes, and the existing disturbances. Command durations are sampled uniformly from 3–8 seconds; consecutive commands may be identical. Each environment samples independently. Stages 1–5 keep zero yaw commands.

In Stages 4–5, first choose stop with 30% probability; otherwise choose one of the four body-frame directions uniformly (25% each among moving commands). Each moving command activates exactly one planar axis, so there are no diagonal commands. Variable speed magnitudes are sampled uniformly within their ranges. Stages 1–3 keep their existing command behavior.

In Stages 6–7, 30% of commands are a full stop: `vx = vy = yaw_rate = 0`. For the other 70%, uniformly choose forward, backward, left, right, or zero translation (20% each), then sample yaw independently. Nonzero planar speed is uniform from 0.3–1 m/s; zero translation gives an in-place turn. Stage 6 chooses either yaw sign with equal probability at a fixed magnitude of 0.5 rad/s. Stage 7 uses a uniform signed yaw range of −0.7–0.7 rad/s. Translation and yaw share the sampled command duration, so the robot can move and turn together.

## Stability Rule

After every training iteration, compare the latest 50 iterations with the preceding 50 from the same stage. Both mean reward and mean value loss must change by less than 5%:

    change(X) = abs(mean_recent_50(X) - mean_previous_50(X))
                / max(abs(mean_previous_50(X)), 1e-8)

Reward is the mean total environment reward over all rollout steps and parallel environments in one iteration. Value loss is the PPO update result, averaged across GPUs when applicable. Policy/surrogate loss is logged but does not gate progression. Non-finite windows cannot qualify.

There are no survival or velocity-error gates and no frozen-model checks. Each visited stage trains for at least 100 iterations. When stable, save a stage checkpoint, print the stage results and checkpoint path, reset all environments and stability history, and immediately train the next stage. Stage 7 stability ends training. Reaching the iteration limit saves the current model.

## Starting Stage

New and resumed runs default to Stage 1. Add `--agent.start-stage 6` with the existing resume options to continue a model trained through Stage 5 with yaw commands. Valid starting stages are `1`–`7`. The selected stage starts with fresh stability history; loaded model and optimizer parameters are retained. Later stages advance automatically. Play uses the stage saved in the checkpoint.

## Stage Logs

Each stable stage prints its stage number, iteration, window means/changes, full checkpoint path, and next action to the terminal. These messages are included in `train.log` when training output is redirected. Snapshots are saved as `stage_<stage>_stable_model_<iteration>.pt` in the new run directory. Training continues without waiting for review.

## Interactive Inspection

Run `uv run --no-sync python src/esra_unified_mjlab_simulator/scripts/leo_policy_viewer.py --checkpoint-file "/path/to/model.pt"`. The browser provides random/manual commands, a random-push toggle, and forward/backward/left/right kick buttons. Random commands automatically use the training profile associated with the checkpoint. It reuses training noise, push settings, and episode resets in one environment without training. See the README for controls.

## File Changes

- **env/mdp/curriculum.py**: `make_velocity_command_cfg()` centrally defines all seven profiles and the common stability rule. `VelocityStageCfg.yaw_rate_magnitude` defines fixed/random yaw magnitudes; its `ang_vel_z` property derives signed bounds. `_apply_stage_cfg()` supplies the selected planar/yaw settings to the sampler.
- **env/mdp/command.py**: `AdditionalDirections.include_in_place` optionally adds zero translation. `yaw_rate_magnitude` samples angular magnitude and sign independently of planar motion; `_resample_command()` supports both fixed and random yaw. Full-stop commands still clear all three velocity components. Earlier stages retain their existing sampling.
- **cfg/leo_quadruped/algo_cfg.py**: Adds `VelocityOnPolicyRunnerCfg.start_stage`, exposed as `--agent.start-stage` with default `1`.
- **runner/velocity_runner.py**: Starts training at the selected stage and advances on stability. `_save_stage_checkpoint()` saves stage snapshots and prints their paths and stability results to the training log. Retains the inference-mode reset fix and old checkpoint compatibility.
- **README.md**: Documents the starting-stage option, progression, and checkpoint/log instructions.
- **scripts/leo_policy_viewer.py**: Adds `ViewerVelocityCommand` / `ViewerVelocityCommandCfg` for browser command control, `PushControls` / `install_push_controls()` for gated training pushes and queued manual kicks, and `LeoPolicyViewer` for playback and readouts. Training settings remain in their existing configuration files.

Reward functions, disturbances, and PPO hyperparameters are unchanged. The existing runner handles all seven stages by reading the stage count, and the policy viewer automatically restores the checkpoint's profile. `Curriculum/velocity/` logs the stage, stage iteration count, window means, relative changes, stability, and completion. No commands, tests, or training were run; validation was limited to static inspection.
