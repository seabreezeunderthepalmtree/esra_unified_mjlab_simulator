Problem:

The robot can stand, but its legs are spread too far apart. This posture allows it to stand but makes walking difficult.

Solution:

Add a temporary penalty for excessive hip abduction (`back/front_left/right_hip_roll`, with a joint range of ±30°).

Changes:

1. Added `hip_roll_excess_l2()` to `src/esra_unified_mjlab_simulator/env/mdp/reward.py` to calculate a posture cost for the four hip roll joints. Deviations within 10° of the default angle incur no cost. Beyond that, the excess is divided by 20°, squared, and averaged across the four legs.

```text
excess_angle_i = max(abs(current_angle_i - default_angle_i) - 10°, 0)
cost = mean((excess_angle_i / 20°)²)
penalty = -0.5 × cost
```

2. Added the following to `src/esra_unified_mjlab_simulator/cfg/leo_quadruped/env_cfg.py`:
   - `ENABLE_TEMP_HIP_ROLL_PENALTY = True`: enables the temporary penalty by default. Set it to `False` to disable it for newly created environments.
   - `temp_hip_roll_excess`: when the switch is enabled, this reward term calls `hip_roll_excess_l2()`. For each robot, the function reads the four `hip_roll` angles and returns one cost value, which is then multiplied by −0.5 to produce the penalty.
