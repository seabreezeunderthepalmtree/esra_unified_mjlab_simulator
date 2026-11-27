"""Flat-ground standing qualification followed by walking and standing.

Task settings and per-environment stage progression live here. Generic velocity
sampling remains in ``command.py``; timing uses mjlab's command manager. Training
and play share this curriculum, including the standing qualification stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG, flat

from .command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "StandingToWalkingCommand",
    "StandingToWalkingCommandCfg",
    "make_velocity_command_cfg",
    "make_terrain_cfg",
    "standing_stage_complete",
    "terrain_levels_vel",
]


class StandingToWalkingCommand(UniformVelocityCommand):
    """Promote each environment after consecutive full standing episodes."""

    cfg: StandingToWalkingCommandCfg

    def __init__(self, cfg: StandingToWalkingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.standing_successes = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_start_step = torch.full_like(
            self.standing_successes, env.common_step_counter
        )
        self._standing_episode_steps = math.ceil(cfg.standing_duration_s / env.step_dt)

    @property
    def walking_enabled(self) -> torch.Tensor:
        return self.standing_successes >= self.cfg.required_standing_successes

    @property
    def standing_episode_complete(self) -> torch.Tensor:
        # The runner randomizes episode_length_buf at startup. Count real steps
        # since reset instead, so a shortened first episode cannot earn success.
        elapsed_steps = self._env.common_step_counter - self._episode_start_step
        return ~self.walking_enabled & (elapsed_steps >= self._standing_episode_steps)

    def _apply_standing_stage(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = (~self.walking_enabled).nonzero(as_tuple=False).flatten()
        else:
            env_ids = env_ids[~self.walking_enabled[env_ids]]
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_w[env_ids] = 0.0
        self._sampled_vel_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = True
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self._apply_standing_stage(env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        standing_ids = env_ids[~self.walking_enabled[env_ids]]
        completed = self.standing_episode_complete[standing_ids]
        failed = self._env.termination_manager.terminated[standing_ids].clone()
        if "out_of_terrain_bounds" in self._env.termination_manager.active_terms:
            failed |= self._env.termination_manager.get_term(
                "out_of_terrain_bounds"
            )[standing_ids]
        streak = self.standing_successes[standing_ids]
        next_streak = torch.where(completed & ~failed, streak + 1, 0)
        # The runner's randomized first timeout is neither a full success nor a
        # physical failure. Ignore it, including after restoring a partial streak.
        if "time_out" in self._env.termination_manager.active_terms:
            shortened_timeout = (
                ~completed
                & ~failed
                & self._env.termination_manager.get_term("time_out")[standing_ids]
            )
            next_streak = torch.where(shortened_timeout, streak, next_streak)
        self.standing_successes[standing_ids] = next_streak
        self._episode_start_step[env_ids] = self._env.common_step_counter
        # Promotion happens before the base reset samples the next command.
        return super().reset(env_ids)

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        super().compute(dt, env_ids)
        # Manual controls become available after standing qualification, so a
        # nonzero manual command cannot count as a successful standing episode.
        self._apply_standing_stage()
        log = self._env.extras.setdefault("log", {})
        log["Curriculum/standing_to_walking/walking_fraction"] = (
            self.walking_enabled.float().mean()
        )
        log["Curriculum/standing_to_walking/success_streak_mean"] = (
            self.standing_successes.float().mean()
        )

    def curriculum_state_dict(self) -> dict[str, torch.Tensor]:
        return {"standing_successes": self.standing_successes.detach().cpu().clone()}

    def load_curriculum_state_dict(
        self, state: dict[str, torch.Tensor] | None
    ) -> None:
        self.standing_successes.zero_()
        if state is not None:
            successes = state["standing_successes"].to(
                device=self.device, dtype=torch.long
            ).flatten()
            # Restore matching env indices; extra new envs start with standing.
            count = min(self.num_envs, successes.numel())
            self.standing_successes[:count] = successes[:count].clamp(
                min=0, max=self.cfg.required_standing_successes
            )
        # Checkpoints restore policy and curriculum progress, not physical episode
        # state. Start a fresh qualification timer at the restored global step.
        self._episode_start_step.fill_(self._env.common_step_counter)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample(env_ids)
        self._update_command(env_ids)


@dataclass(kw_only=True)
class StandingToWalkingCommandCfg(UniformVelocityCommandCfg):
    """Explicit stage settings; the generic command config is not replaced."""

    standing_duration_s: float
    required_standing_successes: int

    def build(self, env: ManagerBasedRlEnv) -> StandingToWalkingCommand:
        return StandingToWalkingCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.standing_duration_s <= 0.0 or self.required_standing_successes < 1:
            raise ValueError("Standing duration and required successes must be positive")


def make_velocity_command_cfg() -> StandingToWalkingCommandCfg:
    """Configure qualification and the subsequent walking/standing commands.

    Only qualified envs use the walking ranges and standing probability. Each
    command holding time is sampled uniformly by mjlab's command manager.
    """
    return StandingToWalkingCommandCfg(
        entity_name="robot",
        standing_duration_s=20.0,
        required_standing_successes=10,
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        rel_world_envs=0.0,
        init_velocity_prob=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.3),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


def standing_stage_complete(
    env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
    """End a full qualification episode as a timeout in both training and play."""
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, StandingToWalkingCommand)
    return command.standing_episode_complete


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
