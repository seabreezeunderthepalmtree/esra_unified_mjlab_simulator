"""Task settings for the current flat-ground walking stage.

Define command ranges, standing probability, command duration, and terrain here.
Training and play build fresh configurations from the same factories. Command
sampling and timing are handled by ``command.py`` and mjlab's command manager.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG, flat

from .command import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "make_velocity_command_cfg",
    "make_terrain_cfg",
    "terrain_levels_vel",
]


def make_velocity_command_cfg() -> UniformVelocityCommandCfg:
    """Configure walking or standing independently for each environment.

    Each resample selects standing with the configured probability; otherwise it
    samples from the velocity ranges. The command manager uniformly samples a
    holding time in ``resampling_time_range``. An episode reset may end it early.
    """
    return UniformVelocityCommandCfg(
        entity_name="robot",
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
