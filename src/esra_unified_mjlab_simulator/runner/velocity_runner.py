import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils import WandbLogWriter, check_nan

from esra_unified_mjlab_simulator.env.mdp.curriculum import VelocityCurriculumCommand


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    @torch.inference_mode()
    def _start_stage(self, command: VelocityCurriculumCommand, stage: int):
        # Rollouts can replace environment buffers with inference tensors.
        # Keep every stage reset in inference mode as well.
        command.begin_stage(stage)
        obs, _ = self.env.reset()
        dones = torch.ones(self.env.num_envs, device=self.device, dtype=torch.long)
        self.alg.get_policy().reset(dones)
        self.alg.critic.reset(dones)
        self.alg.storage.clear()
        self.logger.cur_reward_sum.zero_()
        self.logger.cur_episode_length.zero_()
        self.logger.rewbuffer.clear()
        self.logger.lenbuffer.clear()
        self.logger.ep_extras.clear()
        return obs.to(self.device)

    def _iteration_statistics(self, mean_reward: float, value_loss: float) -> tuple[float, float]:
        values = torch.tensor([mean_reward, value_loss], dtype=torch.float64, device=self.device)
        if self.is_distributed:
            torch.distributed.all_reduce(values)
            values /= self.gpu_world_size
        reward, loss = values.tolist()
        return reward, loss

    def _write_curriculum_metrics(
        self, command: VelocityCurriculumCommand, mean_reward: float, value_loss: float
    ) -> None:
        if self.logger.writer is not None:
            metrics = {
                **command.progress_metrics(),
                "Curriculum/velocity/iteration_mean_reward": mean_reward,
                "Curriculum/velocity/iteration_value_loss": value_loss,
            }
            for name, value in metrics.items():
                self.logger.writer.add_scalar(name, value, self.current_learning_iteration)

    def _save_stage_alert(self, command: VelocityCurriculumCommand) -> None:
        if self.gpu_global_rank != 0:
            return
        if self.logger.log_dir is None:
            raise ValueError("A training log directory is required to save stage checkpoints")
        run_dir = Path(self.logger.log_dir).resolve()
        checkpoint = run_dir / (
            f"stage_{command.stage}_stable_model_{self.current_learning_iteration}.pt"
        )
        self.save(str(checkpoint))
        metrics = command.stability.window_metrics
        next_action = (
            "The final stage is stable; training is complete."
            if command.completed else f"Automatically continuing to Stage {command.stage + 1}."
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        message = (
            f"[{timestamp}] Stage {command.stage} stable at iteration "
            f"{self.current_learning_iteration}.\n"
            f"Reward means: {metrics['reward_previous_mean']:.6g} -> "
            f"{metrics['reward_recent_mean']:.6g}; "
            f"change={metrics['reward_relative_change']:.2%}.\n"
            f"Value loss means: {metrics['value_loss_previous_mean']:.6g} -> "
            f"{metrics['value_loss_recent_mean']:.6g}; "
            f"change={metrics['value_loss_relative_change']:.2%}.\n"
            f"Checkpoint: {checkpoint}\n{next_action}"
        )
        alert_path = Path(__file__).resolve().parents[3] / "alert"
        try:
            with alert_path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n\n")
        except OSError as error:
            print(f"[WARN] Could not append to {alert_path}: {error}", flush=True)
        print(f"[Curriculum] {message}", flush=True)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Train from the selected stage and advance on reward/value-loss stability."""
        command = self.env.unwrapped.command_manager.get_term("twist")
        if not isinstance(command, VelocityCurriculumCommand):
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        self.alg.train_mode()
        # New and resumed runs use the configured starting stage. Loaded weights
        # and optimizer remain intact; stage counters and stability windows reset.
        obs = self._start_stage(command, self.cfg["start_stage"])
        if self.gpu_global_rank == 0:
            print(f"[Curriculum] Starting training at Stage {command.stage}.", flush=True)
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.perf_counter()
            reward_sum = torch.zeros((), device=self.device)
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    )
                    reward_sum += rewards.mean()
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"].get("rnd_cfg") else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                collect_time = time.perf_counter() - start
                self.alg.compute_returns(obs)

            start = time.perf_counter()
            loss_dict = self.alg.update()
            learn_time = time.perf_counter() - start
            self.current_learning_iteration = it
            mean_reward, value_loss = self._iteration_statistics(
                reward_sum.item() / self.cfg["num_steps_per_env"], loss_dict["value"]
            )
            command.stability.record(mean_reward, value_loss)
            if command.stability.stable and command.stage == len(command.cfg.stages):
                command.completed = True
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )
            self._write_curriculum_metrics(command, mean_reward, value_loss)

            if command.stability.stable:
                self._save_stage_alert(command)
                if command.completed:
                    break
                obs = self._start_stage(command, command.stage + 1)
            elif self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(str(Path(self.logger.log_dir) / f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos=None):
        command = self.env.unwrapped.command_manager.get_term("twist")
        if isinstance(command, VelocityCurriculumCommand):
            infos = {
                **(infos or {}),
                "velocity_curriculum": command.curriculum_state_dict(),
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
        if isinstance(command, VelocityCurriculumCommand):
            state = (
                infos.get("velocity_curriculum", infos.get("standing_to_walking"))
                if infos else None
            )
            command.load_curriculum_state_dict(state)
        return infos
