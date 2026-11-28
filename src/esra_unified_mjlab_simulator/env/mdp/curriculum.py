"""Flat-ground command stages and a shared training-stability rule.

All task settings live in make_velocity_command_cfg(). The runner supplies one
reward/value-loss pair per iteration and handles checkpoints and training logs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG, flat

from .command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "VelocityStageCfg",
    "StageStability",
    "VelocityCurriculumCommand",
    "VelocityCurriculumCommandCfg",
    "make_velocity_command_cfg",
    "make_terrain_cfg",
    "curriculum_episode_complete",
    "terrain_levels_vel",
]


@dataclass(frozen=True, kw_only=True)
class VelocityStageCfg:
    lin_vel_x: tuple[float, float]
    standing_probability: float
    additional_directions: UniformVelocityCommandCfg.AdditionalDirections | None = None
    yaw_rate_magnitude: tuple[float, float] | None = None

    @property
    def ang_vel_z(self) -> tuple[float, float]:
        """Derive signed yaw bounds from the stage's magnitude range."""
        if self.yaw_rate_magnitude is None:
            return (0.0, 0.0)
        maximum = self.yaw_rate_magnitude[1]
        return (-maximum, maximum)


class StageStability:
    """Compare adjacent, equally sized windows from the current stage only."""

    def __init__(self, cfg: VelocityCurriculumCommandCfg):
        self.window_size = cfg.stability_window_iterations
        self.threshold = cfg.stability_change_threshold
        self.epsilon = cfg.stability_epsilon
        self._history: deque[tuple[float, float]] = deque(maxlen=2 * self.window_size)
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        self.iterations = 0
        self.stable = False
        self.window_metrics: dict[str, float] = {}

    def record(self, mean_reward: float, value_loss: float) -> None:
        self._history.append((mean_reward, value_loss))
        self.iterations += 1
        self.stable = False
        self.window_metrics = {}
        if len(self._history) < 2 * self.window_size:
            return
        history = list(self._history)
        if not all(math.isfinite(value) for row in history for value in row):
            return
        changes = []
        for column, name in enumerate(("reward", "value_loss")):
            previous = sum(row[column] for row in history[:self.window_size]) / self.window_size
            recent = sum(row[column] for row in history[self.window_size:]) / self.window_size
            change = abs(recent - previous) / max(abs(previous), self.epsilon)
            self.window_metrics[f"{name}_previous_mean"] = previous
            self.window_metrics[f"{name}_recent_mean"] = recent
            self.window_metrics[f"{name}_relative_change"] = change
            changes.append(change)
        self.stable = all(change < self.threshold for change in changes)

    def metrics(self) -> dict[str, float]:
        return {
            "stage_iterations": float(self.iterations),
            "window_ready": float(bool(self.window_metrics)),
            "stable": float(self.stable),
            **self.window_metrics,
        }


class VelocityCurriculumCommand(UniformVelocityCommand):
    """Select a command profile; stability, not episode performance, advances it."""

    cfg: VelocityCurriculumCommandCfg

    def __init__(self, cfg: VelocityCurriculumCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.stage = 1
        self.completed = False
        self.stability = StageStability(cfg)
        self._episode_steps = math.ceil(cfg.episode_duration_s / env.step_dt)
        self._episode_start_step = torch.full(
            (self.num_envs,), env.common_step_counter, dtype=torch.long, device=self.device
        )
        self._apply_stage_cfg()

    @property
    def stage_cfg(self) -> VelocityStageCfg:
        return self.cfg.stages[self.stage - 1]

    def _apply_stage_cfg(self) -> None:
        additional = self.stage_cfg.additional_directions
        lateral_range = (
            (0.0, 0.0)
            if additional is None
            else (-additional.lateral[1], additional.lateral[1])
        )
        self.cfg.ranges = replace(
            self.cfg.ranges,
            lin_vel_x=self.stage_cfg.lin_vel_x,
            lin_vel_y=lateral_range,
            ang_vel_z=self.stage_cfg.ang_vel_z,
        )
        self.cfg.additional_directions = additional
        self.cfg.yaw_rate_magnitude = self.stage_cfg.yaw_rate_magnitude
        self.cfg.rel_standing_envs = self.stage_cfg.standing_probability

    def begin_stage(self, stage: int) -> None:
        if not 1 <= stage <= len(self.cfg.stages):
            raise ValueError(f"Invalid velocity curriculum stage: {stage}")
        self.stage = stage
        self._apply_stage_cfg()
        self.completed = False
        self.stability.reset()
        self._episode_start_step.fill_(self._env.common_step_counter)

    def episode_complete(self) -> torch.Tensor:
        return (
            self._env.common_step_counter - self._episode_start_step
        ) >= self._episode_steps

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        self._episode_start_step[env_ids] = self._env.common_step_counter
        return super().reset(env_ids)

    def progress_metrics(self) -> dict[str, float]:
        values = {
            "stage": float(self.stage),
            "completed": float(self.completed),
            **self.stability.metrics(),
        }
        return {f"Curriculum/velocity/{name}": value for name, value in values.items()}

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        super().compute(dt, env_ids)
        if self.stage == 1:
            # Stage 1 remains a zero-command task even with viewer controls.
            self.vel_command_b.zero_()
            self.vel_command_w.zero_()

    def curriculum_state_dict(self) -> dict[str, int | bool]:
        # Play restores the saved profile. A new learn() starts at the configured
        # stage with fresh stability history and loaded model/optimizer parameters.
        return {"stage": self.stage, "completed": self.completed}

    @torch.inference_mode()
    def load_curriculum_state_dict(self, state: dict | None) -> None:
        stage = 1
        if state is not None:
            if "stage" in state:
                stage = int(state["stage"])
            elif "standing_successes" in state:
                # Legacy metadata used 10 standing successes. This migration only
                # selects a play profile; it is not a training progression gate.
                successes = torch.as_tensor(state["standing_successes"])
                if successes.numel() and successes.max().item() >= 10:
                    stage = 2
        self.begin_stage(stage)
        self.completed = bool(state.get("completed", False)) if state else False
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample(env_ids)
        self._update_command(env_ids)


@dataclass(kw_only=True)
class VelocityCurriculumCommandCfg(UniformVelocityCommandCfg):
    episode_duration_s: float
    stability_window_iterations: int
    stability_change_threshold: float
    stability_epsilon: float
    stages: tuple[VelocityStageCfg, ...]

    def build(self, env: ManagerBasedRlEnv) -> VelocityCurriculumCommand:
        return VelocityCurriculumCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.episode_duration_s <= 0 or self.stability_window_iterations < 1:
            raise ValueError("Episode duration and stability window must be positive")
        if self.stability_change_threshold <= 0 or self.stability_epsilon <= 0:
            raise ValueError("Stability threshold and numerical floor must be positive")
        if not self.stages:
            raise ValueError("At least one velocity curriculum stage is required")
        for stage in self.stages:
            if not 0 <= stage.standing_probability <= 1:
                raise ValueError("Standing probability must be in [0, 1]")
            if not 0 <= stage.lin_vel_x[0] <= stage.lin_vel_x[1]:
                raise ValueError("Forward speed range must be ordered and nonnegative")
            if stage.yaw_rate_magnitude is not None:
                if not 0 <= stage.yaw_rate_magnitude[0] <= stage.yaw_rate_magnitude[1]:
                    raise ValueError("Yaw-rate magnitude range must be ordered and nonnegative")


def make_velocity_command_cfg() -> VelocityCurriculumCommandCfg:
    """The single source for command profiles and the common stability rule."""
    walking_speed = 0.3
    fixed_speed = (walking_speed, walking_speed)
    walking_range = (walking_speed, 1.0)
    walking_stop_probability = 0.3
    fixed_yaw_rate = 0.5
    maximum_yaw_rate = 0.7
    walking_directions = UniformVelocityCommandCfg.AdditionalDirections(
        backward=walking_range, lateral=walking_range
    )
    turning_directions = replace(walking_directions, include_in_place=True)
    stages = (
        VelocityStageCfg(lin_vel_x=(0.0, 0.0), standing_probability=1.0),
        VelocityStageCfg(lin_vel_x=fixed_speed, standing_probability=0.2),
        VelocityStageCfg(
            lin_vel_x=walking_range, standing_probability=walking_stop_probability
        ),
        # Introduce backward/lateral walking while retaining forward speed coverage.
        VelocityStageCfg(
            lin_vel_x=walking_range,
            standing_probability=walking_stop_probability,
            additional_directions=UniformVelocityCommandCfg.AdditionalDirections(
                backward=fixed_speed, lateral=fixed_speed
            ),
        ),
        # Expand the three new directions to the same speed range as forward.
        VelocityStageCfg(
            lin_vel_x=walking_range,
            standing_probability=walking_stop_probability,
            additional_directions=walking_directions,
        ),
        # Combine either fixed yaw direction with any planar command or turn in place.
        VelocityStageCfg(
            lin_vel_x=walking_range,
            standing_probability=walking_stop_probability,
            additional_directions=turning_directions,
            yaw_rate_magnitude=(fixed_yaw_rate, fixed_yaw_rate),
        ),
        # Uniform magnitude and sign together give a uniform signed yaw range.
        VelocityStageCfg(
            lin_vel_x=walking_range,
            standing_probability=walking_stop_probability,
            additional_directions=turning_directions,
            yaw_rate_magnitude=(0.0, maximum_yaw_rate),
        ),
    )
    return VelocityCurriculumCommandCfg(
        entity_name="robot",
        episode_duration_s=20.0,
        stability_window_iterations=50,
        stability_change_threshold=0.05,
        stability_epsilon=1e-8,
        stages=stages,
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=stages[-1].standing_probability,
        additional_directions=stages[-1].additional_directions,
        yaw_rate_magnitude=stages[-1].yaw_rate_magnitude,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        rel_world_envs=0.0,
        init_velocity_prob=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=stages[-1].lin_vel_x,
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=stages[-1].ang_vel_z,
            heading=None,
        ),
    )


def curriculum_episode_complete(
    env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
    """End a full episode in every stage; episode outcomes do not gate progress."""
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, VelocityCurriculumCommand)
    return command.episode_complete()


def make_terrain_cfg() -> TerrainEntityCfg:
    """Create flat terrain without mutating mjlab's shared terrain preset."""
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(
            ROUGH_TERRAINS_CFG,
            sub_terrains={"flat": flat(proportion=1.0)},
            num_rows=1,
            num_cols=1,
        ),
        max_init_terrain_level=0,
    )


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
) -> dict[str, torch.Tensor]:
    """Keep the current flat patches without promoting or demoting environments."""
    del env_ids, command_name
    terrain = env.scene.terrain
    assert terrain is not None
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}
