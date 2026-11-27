# Run 4: From Standing to Flat-Ground Walking

The standing posture is now acceptable. Each environment first completes a standing qualification stage, then learns to walk forward, stop, and start again.

## Stage 1: Standing Qualification

- Use zero forward, lateral, and turning velocity commands.
- A successful episode lasts a full **20 seconds** without falling, illegal ground contact, or leaving the terrain.
- After **10 consecutive successful episodes**, that environment enters Stage 2. These are completed episodes, not training iterations or successes summed across all environments.
- A failed episode or an early manual reset clears that environment's success streak.
- Measure duration using actual simulation steps since reset. Randomized initial episode counters do not count as time spent standing.
- An artificially shortened startup timeout is ignored: it neither earns success nor clears an existing streak restored from a checkpoint.
- Completing a standing episode ends that episode as a timeout. It does not stop the training run.
- Once an environment qualifies, it stays in Stage 2 even if it later falls.

## Stage 2: Walking and Standing

Qualified environments independently sample commands:

- **80% probability:** walk forward at 0.3 m/s, with zero lateral and turning velocity commands.
- **20% probability:** stand still with a zero-velocity command.
- Each command lasts for a duration sampled uniformly between **3 and 8 seconds**, after which both the command and duration are resampled.
- Consecutive samples may produce the same command.
- A fall or episode reset may end the current command early.

## Shared Configuration and Progress

All stage settings are defined once in `curriculum.py`, inside `make_velocity_command_cfg()`:

| Setting | Value |
|---|---|
| `standing_duration_s` | `20.0` seconds |
| `required_standing_successes` | `10` consecutive successful episodes |
| `ranges.lin_vel_x` | `(0.3, 0.3)` m/s |
| `ranges.lin_vel_y` | `(0.0, 0.0)` m/s |
| `ranges.ang_vel_z` | `(0.0, 0.0)` rad/s |
| `rel_standing_envs` | `0.2`, applied after qualification |
| `resampling_time_range` | `(3.0, 8.0)` seconds, sampled uniformly |

There are no new module-level constants. Stage counters and episode start steps are runtime state, not additional configuration values.

Training and play use the same qualification and command rules. Play keeps its existing long episode limit, but the qualification timeout still ends standing episodes after 20 seconds. Manual velocity controls are available only after the selected environment qualifies.

Checkpoints save each environment's completed success streak. Resuming restores that progress and starts a fresh episode timer; it does not count time from the interrupted physical episode. Older checkpoints without this state begin at Stage 1. If the environment count changes, matching indices are restored and extra new environments begin at Stage 1.

Two metrics report progress: `Curriculum/standing_to_walking/walking_fraction` and `Curriculum/standing_to_walking/success_streak_mean`.

## Code Changes

**[env/mdp/curriculum.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/env/mdp/curriculum.py>)**

Added:

- `StandingToWalkingCommand`: extends the existing velocity command with per-environment qualification. Its `reset()` updates success streaks, `_resample_command()` applies the current stage, and `compute()` enforces zero commands during qualification and logs progress. `curriculum_state_dict()` and `load_curriculum_state_dict()` save and restore completed progress.
- `StandingToWalkingCommandCfg`: explicitly extends the generic config with `standing_duration_s` and `required_standing_successes`. Its `build()` creates the staged command. The original `UniformVelocityCommandCfg` class keeps its original name and behavior.
- `make_velocity_command_cfg()`: defines both qualification settings and the subsequent walking/standing command settings in one place.
- `make_terrain_cfg()`: creates the flat-ground terrain configuration.
- `standing_stage_complete()`: detects a full qualification episode. It is registered as a timeout, so successful standing episodes can end in both training and play.

The staged command stores `standing_successes` and `_episode_start_step` for each environment. `walking_enabled` is derived from the success streak; `standing_episode_complete` compares actual elapsed steps against the configured duration. Failures take precedence when success and failure conditions occur on the same step.

Terrain settings are created from a copy of the mjlab preset. Importing this module no longer changes mjlab's global terrain configuration.

Removed the previous standing overrides:

- `_set_standing_command()`, which forced zero velocity and a 100% standing probability.
- The local `UniformVelocityCommandCfg` subclass, which overwrote configuration values during initialization. The generic class from `command.py` is kept unchanged; the new staged subclass has the explicit name `StandingToWalkingCommandCfg`.
- `commands_vel()`, which overwrote the configuration and cleared commands for all environments during resets.

**[env/velocity_env.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/env/velocity_env.py>)**

Updated `make_velocity_env_cfg()`:

- Imports `make_velocity_command_cfg()` and `make_terrain_cfg()`.
- Creates `velocity_command_cfg = make_velocity_command_cfg()` and registers it as `"twist"` in `Commands`, replacing the inline command settings.
- Uses `terrain=make_terrain_cfg()` in `SceneCfg`, replacing the inline terrain settings.
- Removes the `"command_vel"` curriculum entry and its `velocity_stages`. This stage does not automatically increase speed.
- Registers `"standing_stage_complete"` in `Terminations` with `time_out=True`.
- Reads the normal training episode length from `velocity_command_cfg.standing_duration_s`, keeping the 20-second value in one configuration location.
- Removes unused imports and updates the module description.

No new functions or classes are added to this file. It assembles the environment; stage decisions remain in `curriculum.py`.

**[env/mdp/command.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/env/mdp/command.py>)**

Updated the existing `UniformVelocityCommand` class:

| Method | Change |
|---|---|
| `__init__()` | Adds `_sampled_vel_command_b` to store each environment’s latest automatic command. This is runtime state, not a configuration constant. |
| `_resample_command()` | Removes the `.clamp(min=0.3)` minimum forward-speed restriction and saves newly sampled commands. |
| `_update_command()` | Restores automatic commands before applying updates, preventing manual slider values from persisting after manual control is disabled. |

These earlier cleanup changes are retained. The new qualification stage is implemented by the explicitly named subclass in `curriculum.py`; no success thresholds or stage-specific speeds are added to the generic command class.

**[cfg/leo_quadruped/env_cfg.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/cfg/leo_quadruped/env_cfg.py>)**

Updated the `if play:` branch of `make_ccbr_leo_rough_env_cfg()`:

- Removes the overrides that set forward and lateral velocity ranges to zero.
- Removes the override that shortened command duration to 1–3 seconds.
- Removes the two associated assertions and adds a comment explaining that training and play share the same command configuration.
- Updates the play-mode comment to explain that qualification remains active and manual velocity controls become available after qualification. The existing `cfg.curriculum = {}` disables reset-manager curriculum terms in play, but does not disable the staged command itself.

No new functions, classes, or constants were added to this file.

**[runner/velocity_runner.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/runner/velocity_runner.py>)**

Updated `VelocityOnPolicyRunner`:

- `save()`: adds the command's qualification progress to checkpoint metadata under `standing_to_walking`.
- Adds `load()`: restores that progress after loading the model and global step counter, then starts fresh command timers. Older checkpoints remain supported.

No new classes or constants were added to this file.

## Validation

Only static file inspection was performed. No commands, tests, or training were run. Walking performance and qualification behavior still require runtime validation.

