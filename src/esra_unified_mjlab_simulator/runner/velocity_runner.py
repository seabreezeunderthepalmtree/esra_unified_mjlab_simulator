import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils import WandbLogWriter

from esra_unified_mjlab_simulator.env.mdp.curriculum import StandingToWalkingCommand


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        command = self.env.unwrapped.command_manager.get_term("twist")
        if isinstance(command, StandingToWalkingCommand):
            infos = {
                **(infos or {}),
                "standing_to_walking": command.curriculum_state_dict(),
            }
        super().save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            is_wandb = isinstance(self.logger.writer, WandbLogWriter)
            run = wandb.run if is_wandb else None
            run_name = run.name if run is not None and run.name is not None else "local"
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if is_wandb and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        command = self.env.unwrapped.command_manager.get_term("twist")
        if isinstance(command, StandingToWalkingCommand):
            state = infos.get("standing_to_walking") if infos else None
            command.load_curriculum_state_dict(state)
        return infos
