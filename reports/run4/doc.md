# Run 4: From Standing to Flat-Ground Walking

The standing posture is now acceptable. This stage focuses on learning to walk forward, stop, and start again.

Training remains on flat ground. Each environment independently samples commands:

- **80% probability:** walk forward at 0.3 m/s, with zero lateral and turning velocity commands.
- **20% probability:** stand still with a zero-velocity command.
- Each command lasts for a duration sampled uniformly between **3 and 8 seconds**, after which both the command and duration are resampled.
- Consecutive samples may produce the same command.
- A fall or episode reset may end the current command early.

## Code Changes

**[env/mdp/curriculum.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/env/mdp/curriculum.py>)**

Added:

- `make_velocity_command_cfg()`: defines the velocity ranges, standing probability, and command duration.
- `make_terrain_cfg()`: creates the flat-ground terrain configuration.

Removed the previous standing overrides:

- `_set_standing_command()`, which forced zero velocity and a 100% standing probability.
- The local `UniformVelocityCommandCfg` subclass, which overwrote configuration values during initialization. The existing class from `command.py` is now used directly.
- `commands_vel()`, which overwrote the configuration and cleared commands for all environments during resets.

**[env/velocity_env.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/env/velocity_env.py>)**

Updated `make_velocity_env_cfg()`:

- Imports `make_velocity_command_cfg()` and `make_terrain_cfg()`.
- Uses `"twist": make_velocity_command_cfg()` in `Commands`, replacing the inline command settings.
- Uses `terrain=make_terrain_cfg()` in `SceneCfg`, replacing the inline terrain settings.
- Removes the `"command_vel"` curriculum entry and its `velocity_stages`. This stage does not automatically increase speed.
- Removes unused imports and updates the module description.

**[env/mdp/command.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/env/mdp/command.py>)**

Updated the existing `UniformVelocityCommand` class:

| Method | Change |
|---|---|
| `__init__()` | Adds `_sampled_vel_command_b` to store each environment’s latest automatic command. This is runtime state, not a configuration constant. |
| `_resample_command()` | Removes the `.clamp(min=0.3)` minimum forward-speed restriction and saves newly sampled commands. |
| `_update_command()` | Restores automatic commands before applying updates, preventing manual slider values from persisting after manual control is disabled. |

**[cfg/leo_quadruped/env_cfg.py](</Users/tianweishi/Documents/课内/Columbia/CURC/esra_unified_mjlab_simulator/src/esra_unified_mjlab_simulator/cfg/leo_quadruped/env_cfg.py>)**

Updated the `if play:` branch of `make_ccbr_leo_rough_env_cfg()`:

- Removes the overrides that set forward and lateral velocity ranges to zero.
- Removes the override that shortened command duration to 1–3 seconds.
- Removes the two associated assertions and adds a comment explaining that training and play share the same command configuration.

No new functions, classes, or constants were added to this file.


