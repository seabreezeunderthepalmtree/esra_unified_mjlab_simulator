"""Inspect a Leo checkpoint with training randomization and interactive controls."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

from esra_unified_mjlab_simulator.cfg.leo_quadruped.algo_cfg import (
    ccbr_leo_ppo_runner_cfg,
)
from esra_unified_mjlab_simulator.cfg.leo_quadruped.env_cfg import (
    make_ccbr_leo_rough_env_cfg,
)
from esra_unified_mjlab_simulator.env.mdp.curriculum import (
    VelocityCurriculumCommand,
    VelocityCurriculumCommandCfg,
)
from esra_unified_mjlab_simulator.runner import VelocityOnPolicyRunner


@dataclass
class PushControls:
    random_enabled: bool = True
    pending: SimpleQueue[tuple[float, float]] = field(default_factory=SimpleQueue)

    def drain(self) -> tuple[float, float]:
        """Combine queued body-frame kicks; only the simulation thread drains them."""
        dx = dy = 0.0
        while True:
            try:
                x, y = self.pending.get_nowait()
            except Empty:
                return dx, dy
            dx += x
            dy += y


class ViewerVelocityCommand(VelocityCurriculumCommand):
    """Keep the training sampler and add a viewer-only manual override."""

    def __init__(self, cfg: VelocityCurriculumCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._mode: viser.GuiDropdownHandle | None = None
        self._manual_inputs: list[viser.GuiNumberHandle] = []

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del get_env_idx, on_change, request_action  # Exactly one environment.
        with server.gui.add_folder(name.capitalize()):
            self._mode = server.gui.add_dropdown(
                "Command mode", options=("Training random", "Manual")
            )
            self._manual_inputs = [
                server.gui.add_number(label, initial_value=0.0, step=0.05, disabled=True)
                for label in (
                    "Forward velocity (m/s)", "Left velocity (m/s)", "Yaw rate (rad/s)"
                )
            ]

            @self._mode.on_update
            def _mode_changed(_event) -> None:
                for control in self._manual_inputs:
                    control.disabled = self._mode.value != "Manual"

            stop = server.gui.add_button("Stop / zero command")

            @stop.on_click
            def _stop(_event) -> None:
                for control in self._manual_inputs:
                    control.value = 0.0
                    control.disabled = False
                self._mode.value = "Manual"

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        super().compute(dt, env_ids)
        if self._mode is not None and self._mode.value == "Manual":
            values = [float(control.value) for control in self._manual_inputs]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Manual velocity commands must be finite")
            # Apply the override before observations are cached.
            self.vel_command_b[:] = self.vel_command_b.new_tensor(values)


@dataclass(kw_only=True)
class ViewerVelocityCommandCfg(VelocityCurriculumCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> ViewerVelocityCommand:
        return ViewerVelocityCommand(self, env)


def install_push_controls(cfg: ManagerBasedRlEnvCfg, controls: PushControls) -> float:
    """Wrap the existing push event; keep its distribution, timer, and parameters."""
    training_push = cfg.events["push_robot"]
    push_function = training_push.func

    def random_push(env, env_ids, **params) -> None:
        if controls.random_enabled:
            push_function(env, env_ids, **params)

    def manual_push(env, env_ids, **params) -> None:
        del env_ids
        dx, dy = controls.drain()
        if dx == 0.0 and dy == 0.0:
            return
        heading = float(env.scene["robot"].data.heading_w[0])
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        dx_w, dy_w = cos_h * dx - sin_h * dy, sin_h * dx + cos_h * dy
        params["velocity_range"] = {"x": (dx_w, dx_w), "y": (dy_w, dy_w)}
        ids = torch.tensor([0], dtype=torch.long, device=env.device)
        push_function(env, ids, **params)
        # Interval events follow step events. Refresh so a random push in this
        # same step adds to this kick rather than reading pre-kick velocity.
        env.sim.forward()

    # Closures keep the same UI state through EventManager's configuration copy.
    cfg.events["push_robot"] = replace(training_push, func=random_push)
    cfg.events["viewer_manual_push"] = EventTermCfg(
        func=manual_push, mode="step", params=dict(training_push.params)
    )
    ranges = training_push.params["velocity_range"]
    return max(abs(value) for axis in ("x", "y") for value in ranges[axis])


class LeoPolicyViewer(ViserPlayViewer):
    def __init__(self, env, policy, server, pushes: PushControls, push_size: float):
        super().__init__(env, policy, viser_server=server)
        self.pushes = pushes
        self.push_size = push_size
        self._last_readout = 0.0

    def setup(self) -> None:
        super().setup()
        with self._server.gui.add_folder("Push controls"):
            random = self._server.gui.add_checkbox(
                "Training random pushes", initial_value=self.pushes.random_enabled
            )

            @random.on_update
            def _random_changed(_event) -> None:
                self.pushes.random_enabled = random.value

            strength = self._server.gui.add_number(
                "Push size (delta-v, m/s)",
                initial_value=self.push_size,
                min=0.0,
                step=0.05,
                hint="Instantaneous velocity kick, using the training push mechanism.",
            )
            buttons = self._server.gui.add_button_group(
                "Push direction", options=("Forward", "Backward", "Left", "Right")
            )

            @buttons.on_click
            def _push(event) -> None:
                size = float(strength.value)
                if not math.isfinite(size):
                    return
                directions = {
                    "Forward": (size, 0.0),
                    "Backward": (-size, 0.0),
                    "Left": (0.0, size),
                    "Right": (0.0, -size),
                }
                # UI callbacks never write simulation tensors.
                self.pushes.pending.put(directions[event.target.value])

            self._server.gui.add_markdown(
                "Directions follow the robot's heading. Pushes also work with "
                "random pushes off. While paused, a button queues a kick for the next step."
            )
        with self._server.gui.add_folder("Velocity readout"):
            units = "Forward/left velocity in m/s; yaw rate in rad/s."
            self._command_text = self._server.gui.add_text(
                "Command (vx, vy, wz)", "", disabled=True, hint=units
            )
            self._actual_text = self._server.gui.add_text(
                "Actual (vx, vy, wz)", "", disabled=True, hint=units
            )

    @torch.inference_mode()
    def _execute_step(self) -> bool:
        # Environment buffers created by rollouts must also reset in inference mode.
        with self._sim_lock:
            return super()._execute_step()

    @torch.inference_mode()
    def reset_environment(self) -> None:
        self.pushes.drain()
        super().reset_environment()

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        if time.monotonic() - self._last_readout < 0.1:
            return
        self._last_readout = time.monotonic()
        command = self.env.unwrapped.command_manager.get_term("twist")
        robot = self.env.unwrapped.scene["robot"]
        target = command.command[0].tolist()
        actual = robot.data.root_link_lin_vel_b[0, :2].tolist()
        actual.append(float(robot.data.root_link_ang_vel_b[0, 2]))
        self._command_text.value = ", ".join(f"{value:.3f}" for value in target)
        self._actual_text.value = ", ".join(f"{value:.3f}" for value in actual)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file", type=Path, required=True, help="Path to a Leo .pt checkpoint."
    )
    parser.add_argument(
        "--device", default=None, help="cpu or cuda:0; defaults to CUDA when available."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    args = parser.parse_args()
    checkpoint = args.checkpoint_file.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint file not found: {checkpoint}")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")

    # Use training settings, including observation noise, resets, and disturbances.
    # The standard play configuration deliberately disables some of these.
    cfg = make_ccbr_leo_rough_env_cfg()
    cfg.scene.num_envs = 1
    command_cfg = cfg.commands["twist"]
    assert isinstance(command_cfg, VelocityCurriculumCommandCfg)
    cfg.commands["twist"] = ViewerVelocityCommandCfg(
        **{item.name: getattr(command_cfg, item.name) for item in fields(command_cfg) if item.init}
    )
    pushes = PushControls()
    push_size = install_push_controls(cfg, pushes)
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    server = None
    try:
        agent_cfg = ccbr_leo_ppo_runner_cfg()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = VelocityOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        with torch.inference_mode():
            wrapped.reset()
        print(f"[Leo] Checkpoint: {checkpoint}")
        print("[Leo] Checkpoint command profile restored; training randomization enabled.")
        server = viser.ViserServer(host=args.host, port=args.port, label="Leo policy viewer")
        LeoPolicyViewer(wrapped, policy, server, pushes, push_size).run()
    finally:
        if server is not None:
            server.stop()
        env.close()


if __name__ == "__main__":
    main()
