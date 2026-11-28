# README

- Install `uv` if you haven't already.

   macOS:
   ```
   brew install uv
   ```
   Linux:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
- Run the following command in the project root to set up the environment based on `pyproject.toml` and `uv.lock`:
   ```
   uv sync --locked --python 3.13
   ```
- Run a Python script from the project root:
   ```bash
   uv run --no-sync python <path/to/script.py>
   ```
- Before training — W&B login:
   Training uses Weights & Biases (W&B) for metric logging by default. If using W&B, log in once in the project root on the machine where training will run. The TensorBoard option below does not require W&B login.
   ```bash
   uv run --no-sync wandb login
   ```
   Create an API key in your [W&B account settings](https://wandb.ai/settings), then paste it into the terminal when prompted. Keep the key private. Without a configured API key, training exits with `No API key configured` before the first training iteration.

   Note that W&B is not available in China. Use TensorBoard instead.
- Training:
   Use the default CUDA GPU:
   ```bash
   uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo
   ```
   For a headless GPU:
   ```bash
   MUJOCO_GL=disable uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo --gpu-ids "[0]" --env.scene.num-envs 4096 --agent.max-iterations 1500 > train.log 2>&1
   ```

   To continue training from an existing checkpoint:
   ```bash
   MUJOCO_GL=disable uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo --gpu-ids "[0]" --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.resume True --agent.load-run "2026-11-28_02-46-13" --agent.load-checkpoint "model_1499.pt" > train.log 2>&1
   ```

   Add `--agent.start-stage 6` (valid values: `1`–`7`, default: `1`) with the resume options to continue a model trained through Stage 5 with yaw commands.

   Stage 6 uses fixed yaw rates of −0.5 or +0.5 rad/s; Stage 7 samples yaw uniformly from −0.7 to +0.7 rad/s. Yaw is independent of the planar command: forward/backward/left/right at 0.3–1 m/s, or zero translation for in-place turns (five equally likely choices outside full stops). Both stages keep 30% full stops (`vx = vy = yaw_rate = 0`), uniform 3–8-second command durations, and the existing stability rule. Stage 7 stability ends training. See [Run 5](reports/run5/doc.md).

- Saved models:
   Checkpoints are saved every 50 iterations and at the end of training:
   ```text
   logs/rsl_rl/ccbr_leo_velocity/<run_timestamp>/model_<iteration>.pt
   ```
   Each stable stage also saves `stage_<stage>_stable_model_<iteration>.pt` in the same run directory and prints its full path and stability results to the terminal (`train.log` when redirected as above). Stage transitions happen automatically without waiting for review.

- Run a trained model:
   ```bash
   uv run --no-sync python src/esra_unified_mjlab_simulator/scripts/leo_policy_viewer.py --checkpoint-file "/path/to/model.pt"
   ```

- Add a new dependency:
   ```
   uv add <package-name>
   ```

`typings/`: Intended for Pylance only; has no effect at runtime.

- `src/esra_unified_mjlab_simulator/`
  - `scripts/`
    - `leo_viewer.py`: View Leo’s body and manually control joint torques or angles. Defaults to **Torque** mode; select **Angle** under **Control** to adjust joint angles.
    - `leo_policy_viewer.py`: Load a policy checkpoint and inspect random/manual commands and random/directional pushes using the training environment.


MUJOCO_GL=disable uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo --gpu-ids "[0]" --env.scene.num-envs 4096 --agent.max-iterations 2000 --agent.resume True --agent.load-run "2026-11-27_21-27-02" --agent.start-stage 4 --agent.load-checkpoint "stage_3_stable_model_1973.pt" > train.log 2>&1
