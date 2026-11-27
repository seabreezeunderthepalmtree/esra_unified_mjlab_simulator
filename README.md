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
- Training:
   Use the default CUDA GPU:
   ```bash
   uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo
   ```
   For a headless GPU:
   ```bash
   MUJOCO_GL=disable uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo --gpu-ids "[0]" --env.scene.num-envs 4096 --agent.max-iterations 1500 > train.log 2>&1
- Saved models:
   Checkpoints are saved every 50 iterations and at the end of training:
   ```text
   logs/rsl_rl/ccbr_leo_velocity/<run_timestamp>/model_<iteration>.pt
   ```
- Run a trained model:
   Replace the checkpoint path with an existing `.pt` file:
   ```bash
   uv run --no-sync play Esra-Velocity-Rough-CCBR-Leo --num-envs 1 --device cpu --viewer viser --checkpoint-file "logs/rsl_rl/ccbr_leo_velocity/<run_timestamp>/model_50.pt"
   ```
   Open the viewer URL printed in the terminal. On a CUDA machine, use `--device cuda:0` instead.
   The current play configuration sets forward and lateral velocity commands to zero; the turning command is unchanged, so the robot may stand or turn rather than walk forward.
- Add a new dependency:
   ```
   uv add <package-name>
   ```

`typings/`: Intended for Pylance only; has no effect at runtime.

- `src/esra_unified_mjlab_simulator/`
  - `scripts/`
    - `leo_viewer.py`: View Leo’s body and manually control joint torques or angles. Defaults to **Torque** mode; select **Angle** under **Control** to adjust joint angles.
