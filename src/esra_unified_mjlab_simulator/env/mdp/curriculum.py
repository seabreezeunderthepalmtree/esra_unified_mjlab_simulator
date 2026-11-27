"""Flat-ground stage settings, command sampling, and episode acceptance.

The runner decides when to check or train. All task values and acceptance rules
live here; command.py remains a generic velocity command implementation.
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
    require_tracking: bool


class VelocityCurriculumCommand(UniformVelocityCommand):
    """One model-wide stage and a window of completed episodes across all envs."""

    cfg: VelocityCurriculumCommandCfg

    def __init__(self, cfg: VelocityCurriculumCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.stage = 1
        self.checking = True
        self.recording = False
        self.passed = False
        self.completed = False
        self._window: deque[tuple[bool, float]] = deque(maxlen=cfg.acceptance_episodes)
        self._pending_results: list[tuple[bool, float]] = []
        self._episode_steps = math.ceil(cfg.episode_duration_s / env.step_dt)
        self._episode_start_step = torch.full(
            (self.num_envs,), env.common_step_counter, dtype=torch.long, device=self.device
        )
        self._error_sum = torch.zeros(self.num_envs, device=self.device)
        self._sample_count = torch.zeros_like(self._episode_start_step)
        self._last_record_step = env.common_step_counter
        self._apply_stage_cfg()

    @property
    def stage_cfg(self) -> VelocityStageCfg:
        return self.cfg.stages[self.stage - 1]

    @property
    def check_finished(self) -> bool:
        return len(self._window) == self.cfg.acceptance_episodes

    @property
    def mean_error(self) -> float:
        return sum(error for _, error in self._window) / max(len(self._window), 1)

    def _apply_stage_cfg(self) -> None:
        # Select a profile here; the inherited sampler still draws each command
        # and standing flag exactly once using the active generic config.
        self.cfg.ranges = replace(self.cfg.ranges, lin_vel_x=self.stage_cfg.lin_vel_x)
        self.cfg.rel_standing_envs = self.stage_cfg.standing_probability

    def begin_stage(self, stage: int, *, checking: bool) -> None:
        """Clear old results before the runner resets every environment."""
        if not 1 <= stage <= len(self.cfg.stages):
            raise ValueError(f"Invalid velocity curriculum stage: {stage}")
        self.stage = stage
        self._apply_stage_cfg()
        self.checking = checking
        self.recording = False
        self.passed = False
        self.completed = False
        self._window.clear()
        self._pending_results.clear()
        self._error_sum.zero_()
        self._sample_count.zero_()
        self._episode_start_step.fill_(self._env.common_step_counter)
        self._last_record_step = self._env.common_step_counter

    def record_step(self) -> torch.Tensor:
        """Measure the held command before termination resets the physical state."""
        step = self._env.common_step_counter
        if self.recording and step != self._last_record_step:
            target = self.vel_command_b[:, :2]
            reference = target.norm(dim=-1).clamp_min(self.cfg.error_reference_speed)
            error = (self.robot.data.root_link_lin_vel_b[:, :2] - target).norm(dim=-1)
            self._error_sum += error / reference
            self._sample_count += 1
        self._last_record_step = step
        return (step - self._episode_start_step) >= self._episode_steps

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        if self.recording:
            # Terminal errors were measured by record_step(), before sim.reset().
            ids = env_ids[self._sample_count[env_ids] > 0].sort().values
            elapsed = self._env.common_step_counter - self._episode_start_step[ids]
            failed = self._env.termination_manager.terminated[ids].clone()
            if "out_of_terrain_bounds" in self._env.termination_manager.active_terms:
                failed |= self._env.termination_manager.get_term("out_of_terrain_bounds")[ids]
            survived = (elapsed >= self._episode_steps) & ~failed
            errors = self._error_sum[ids] / self._sample_count[ids]
            self._pending_results.extend(zip(survived.tolist(), errors.tolist(), strict=True))
        self._error_sum[env_ids] = 0.0
        self._sample_count[env_ids] = 0
        self._episode_start_step[env_ids] = self._env.common_step_counter
        return super().reset(env_ids)

    def pop_episode_results(self) -> list[tuple[bool, float]]:
        results, self._pending_results = self._pending_results, []
        return results

    def record_results(self, results: list[tuple[bool, float]]) -> None:
        """Check the first window in check mode, or each rolling window in training."""
        for result in results:
            if self.passed or (self.checking and self.check_finished):
                break
            self._window.append(result)
            if self.check_finished:
                self.passed = all(survived for survived, _ in self._window) and (
                    not self.stage_cfg.require_tracking
                    or self.mean_error < self.cfg.max_mean_velocity_error
                )

    def progress_metrics(self) -> dict[str, float]:
        count = len(self._window)
        return {
            "Curriculum/velocity/stage": float(self.stage),
            "Curriculum/velocity/checking": float(self.checking),
            "Curriculum/velocity/window_episodes": float(count),
            "Curriculum/velocity/survival_rate": (
                sum(survived for survived, _ in self._window) / max(count, 1)
            ),
            "Curriculum/velocity/mean_relative_error": self.mean_error,
            "Curriculum/velocity/passed": float(self.passed),
            "Curriculum/velocity/completed": float(self.completed),
        }

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        super().compute(dt, env_ids)
        if self.stage == 1:
            # Viewer controls must not turn standing qualification into walking.
            self.vel_command_b.zero_()
            self.vel_command_w.zero_()
        self._env.extras.setdefault("log", {}).update(self.progress_metrics())

    def curriculum_state_dict(self) -> dict[str, int | bool]:
        # A resumed learn() always checks from Stage 1. Save the current stage for
        # play; unfinished physical episodes/windows are not transferable state.
        return {"stage": self.stage, "completed": self.completed}

    def load_curriculum_state_dict(self, state: dict | None) -> None:
        stage = 1
        if state is not None:
            if "stage" in state:
                stage = int(state["stage"])
            elif "standing_successes" in state:
                # Older checkpoints only knew standing -> fixed-speed walking.
                successes = torch.as_tensor(state["standing_successes"])
                if successes.numel() and successes.max().item() >= self.cfg.acceptance_episodes:
                    stage = 2
        self.begin_stage(stage, checking=True)
        self.completed = bool(state.get("completed", False)) if state else False
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample(env_ids)
        self._update_command(env_ids)


@dataclass(kw_only=True)
class VelocityCurriculumCommandCfg(UniformVelocityCommandCfg):
    episode_duration_s: float
    acceptance_episodes: int
    max_mean_velocity_error: float
    error_reference_speed: float
    stages: tuple[VelocityStageCfg, ...]

    def build(self, env: ManagerBasedRlEnv) -> VelocityCurriculumCommand:
        return VelocityCurriculumCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.episode_duration_s <= 0 or self.acceptance_episodes < 1:
            raise ValueError("Episode duration and acceptance window must be positive")
        if self.error_reference_speed <= 0 or self.max_mean_velocity_error <= 0:
            raise ValueError("Error reference speed and threshold must be positive")
        if not self.stages:
            raise ValueError("At least one velocity curriculum stage is required")
        for stage in self.stages:
            if not 0 <= stage.standing_probability <= 1:
                raise ValueError("Standing probability must be in [0, 1]")
            if not 0 <= stage.lin_vel_x[0] <= stage.lin_vel_x[1]:
                raise ValueError("Forward speed range must be ordered and nonnegative")


def make_velocity_command_cfg() -> VelocityCurriculumCommandCfg:
    """The single source for all three stages and their common acceptance rule."""
    walking_speed = 0.3
    stages = (
        VelocityStageCfg(
            lin_vel_x=(0.0, 0.0), standing_probability=1.0, require_tracking=False
        ),
        VelocityStageCfg(
            lin_vel_x=(walking_speed, walking_speed),
            standing_probability=0.2,
            require_tracking=True,
        ),
        VelocityStageCfg(
            lin_vel_x=(walking_speed, 1.0),
            standing_probability=0.3,
            require_tracking=True,
        ),
    )
    return VelocityCurriculumCommandCfg(
        entity_name="robot",
        episode_duration_s=20.0,
        acceptance_episodes=10,
        max_mean_velocity_error=0.05,
        error_reference_speed=walking_speed,
        stages=stages,
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=stages[-1].standing_probability,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        rel_world_envs=0.0,
        init_velocity_prob=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=stages[-1].lin_vel_x,
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


def curriculum_episode_complete(
    env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
    """Record the terminal sample and end a full episode in every stage."""
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, VelocityCurriculumCommand)
    return command.record_step()


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
