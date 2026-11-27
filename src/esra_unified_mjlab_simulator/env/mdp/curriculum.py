"""Fixed flat-ground standing curriculum using the existing environment entry points.

The environment already imports this module through ``mdp``. Before a scene is
built, restrict its shared terrain preset to flat patches and provide a standing
command configuration under the existing name. The reset callbacks keep these
settings fixed; the old velocity stages are deliberately unused. Random pushes
remain entirely under the environment's event configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.terrains.config import ROUGH_TERRAINS_CFG, flat

from .command import UniformVelocityCommand
from .command import UniformVelocityCommandCfg as _VelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["UniformVelocityCommandCfg", "commands_vel", "terrain_levels_vel"]

# velocity_env.py copies this shared preset when constructing each environment.
# Select flat patches before construction, rather than changing an existing scene.
# One training row also clamps the existing max_init_terrain_level to level zero.
ROUGH_TERRAINS_CFG.sub_terrains = {"flat": flat(proportion=1.0)}
ROUGH_TERRAINS_CFG.num_rows = 1
ROUGH_TERRAINS_CFG.num_cols = 1


def _set_standing_command(cfg: _VelocityCommandCfg) -> None:
    cfg.ranges = cfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=None,
    )
    cfg.heading_command = False
    cfg.rel_standing_envs = 1.0
    cfg.rel_heading_envs = 0.0
    cfg.rel_forward_envs = 0.0
    cfg.rel_world_envs = 0.0
    cfg.init_velocity_prob = 0.0


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(_VelocityCommandCfg):
    """Keep the existing config name while selecting standing from the first reset.

    This overrides the walking defaults supplied by velocity_env.py. Applying it
    during config creation also covers play mode, which disables reset curricula.
    """

    def __post_init__(self) -> None:
        _set_standing_command(self)
        super().__post_init__()


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


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    """Keep zero velocity commands regardless of training step or old stages."""
    del env_ids, velocity_stages
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, UniformVelocityCommand)
    _set_standing_command(command.cfg)
    command.vel_command_b.zero_()
    command.vel_command_w.zero_()
    command.is_standing_env.fill_(True)
    command.is_heading_env.zero_()
    command.is_forward_env.zero_()
    command.is_world_env.zero_()
    return {
        f"{axis}_{bound}": torch.tensor(0.0, device=env.device)
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        for bound in ("min", "max")
    }
