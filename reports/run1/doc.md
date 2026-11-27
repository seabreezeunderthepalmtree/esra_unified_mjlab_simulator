I noticed NaN values during training.

edit: `actor` and `critic`s' `obs_normalization` set to `True`

Problem:

| Chart name | Expected | Observed |
|---|---|---|
| `Train/mean_episode_length` | The line should go up as the robot learns to last longer, toward 1,000 steps (20 seconds). | The line stays around 11 steps, meaning the robot fails and starts over after about 0.22 seconds. |
| `Episode_Termination/illegal_contact` | The line should go down toward zero as fewer attempts end because a body part other than a foot touches the ground. | The line stays around 350. The system keeps ending attempts because it detects this kind of contact. |
| `Episode_Termination/time_out` | The line should rise above zero as more attempts last the full 20 seconds. | The line stays near zero. Almost no attempts last the full 20 seconds. |

Possible reasons:

Curriculum is too hard!
**1. Terrain Difficulty**

- Adjusted features: stair height, uphill and downhill slope steepness, height variations on rough terrain, and wave amplitude; flat terrain remains flat.
- Initial state: robots are randomly assigned to **levels 0–5**, rather than all starting at the easiest level.
- Adjustment timing: at the end of each episode, when the robot resets.
- Adjustment criteria: let `D` be the required travel distance calculated from the commands, and `d` be the robot’s actual straight-line distance from its starting point:
  - Increase difficulty: `d > min(80% × D, 4 m)` and `D ≥ 1 m`.
  - Decrease difficulty: `d < 50% × D`, provided the increase condition is not met.
  - Otherwise, keep the current level. If an episode ends early, the required movement for the remaining time is still included.

**2. Velocity Command Ranges**

| Training Progress | Forward/Backward Velocity | Lateral Velocity | Turning Angular Velocity |
|---|---:|---:|---:|
| Start | −1 to 1 m/s | −1 to 1 m/s | −0.5 to 0.5 rad/s |
| 120,000 steps, approximately 5,000 iterations | −1.5 to 2 m/s | Unchanged | −0.7 to 0.7 rad/s |
| 240,000 steps, approximately 10,000 iterations | −2 to 3 m/s | Unchanged | Unchanged |

After a threshold is reached, the ranges are updated when an episode resets, **without checking whether the robot has learned to perform the task**. Specific velocity commands are resampled every **3–8 seconds** within the allowed ranges.

Random pushes, rewards, and contact-based termination conditions are managed by other parts of the code.

Solution: 

Train the robot only on flat ground to remain standing while being pushed.
