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
- Training:
   Use the default CUDA GPU:
   ```bash
   uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo
   ```
   For a short CPU run on macOS:
   ```bash
   uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo \
     --gpu-ids None \
     --env.scene.num-envs 4 \
     --agent.max-iterations 10
   ```
- Saved models:
   Checkpoints are saved every 50 iterations and at the end of training:
   ```text
   logs/rsl_rl/ccbr_leo_velocity/<run_timestamp>/model_<iteration>.pt
   ```
- Training metrics:
   Training can run on a cloud server without a browser. View the charts in your own computer's browser using either option below.

   **Option A — W&B (default):** The server uploads metrics to W&B. Complete any W&B login/API-key setup requested in the server terminal. Open the printed W&B run link in your own computer's browser, signed in to an account with access to the run. The server needs internet access; no SSH port forwarding is needed.

   **Option B — TensorBoard over SSH:** Metrics stay in the server's log directory.

   1. On the **cloud server**, run training from the project root with TensorBoard logging:

      ```bash
      uv run --no-sync train Esra-Velocity-Rough-CCBR-Leo --agent.logger tensorboard
      ```

   2. In another terminal on the **cloud server**, start TensorBoard from the same project root:

      ```bash
      uv run --no-sync tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
      ```

   3. In a terminal on **your own computer**, forward the server's TensorBoard port. Replace `USER` and `SERVER` with your SSH username and server address; add the same SSH port/key options you normally use to connect:

      ```bash
      ssh -N -L 6006:127.0.0.1:6006 USER@SERVER
      ```

      Keep this terminal open. It may show no output while forwarding.

   4. Open **http://localhost:6006** in **your own computer's browser**. SSH carries the connection to TensorBoard on the server. No browser is needed on the server.

   If training and viewing on the same computer, skip the SSH step and open http://localhost:6006 directly.

   If `tensorboard` is not installed in the project environment, install it first:

   ```bash
   uv add tensorboard
   ```

   | Metric | What to check |
   |---|---|
   | `Train/mean_reward` | Total reward should generally increase. |
   | `Train/mean_episode_length` | Longer episodes can indicate fewer falls; check termination reasons too. |
   | `error_vel_xy` | Linear velocity tracking error; lower is better. |
   | `error_vel_yaw` | Turning velocity tracking error; lower is better. |
   | `Episode_Termination/...` | Why episodes end, such as falls, illegal contact, or timeout. |

- Run a trained model:
   Replace the checkpoint path with an existing `.pt` file:
   ```bash
   uv run --no-sync play Esra-Velocity-Rough-CCBR-Leo \
     --checkpoint-file "logs/rsl_rl/ccbr_leo_velocity/<run_timestamp>/model_50.pt" \
     --num-envs 1 \
     --device cpu \
     --viewer viser
   ```
   Open the viewer URL printed in the terminal. On a CUDA machine, use `--device cuda:0` instead.
   The current play configuration sets forward and lateral velocity commands to zero; the turning command is unchanged, so the robot may stand or turn rather than walk forward.
- Add a new dependency:
   ```
   uv add <package-name>
   ```

`typings/`: Intended for Pylance only; has no effect at runtime.

`QUESTIONS.md`: Questions that need to be asked.

- `src/esra_unified_mjlab_simulator/`
  - `scripts/`
    - `leo_viewer.py`: View Leo’s body and manually control joint torques or angles. Defaults to **Torque** mode; select **Angle** under **Control** to adjust joint angles.
