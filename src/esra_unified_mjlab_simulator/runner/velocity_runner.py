import os
import time

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

    def _start_collection(
        self, command: VelocityCurriculumCommand, stage: int, *, checking: bool
    ):
        command.begin_stage(stage, checking=checking)
        obs, _ = self.env.reset()
        dones = torch.ones(self.env.num_envs, device=self.device, dtype=torch.long)
        self.alg.get_policy().reset(dones)
        self.alg.critic.reset(dones)
        self.alg.storage.clear()
        self.logger.cur_reward_sum.zero_()
        self.logger.cur_episode_length.zero_()
        command.recording = True
        return obs.to(self.device)

    def _collect_results(self, command: VelocityCurriculumCommand) -> None:
        results = command.pop_episode_results()
        if self.is_distributed:
            # All ranks participate on every step, even if no local env ended.
            # Order is simulation step, then rank, then local environment index.
            gathered = [None] * self.gpu_world_size
            torch.distributed.all_gather_object(gathered, results)
            results = [result for rank_results in gathered for result in rank_results]
        command.record_results(results)

    def _write_curriculum_metrics(self, command: VelocityCurriculumCommand) -> None:
        if self.logger.writer is not None:
            for name, value in command.progress_metrics().items():
                self.logger.writer.add_scalar(name, value, self.current_learning_iteration)

    def _check_stage(self, command: VelocityCurriculumCommand, stage: int) -> bool:
        """Run the first acceptance window without changing model parameters."""
        obs = self._start_collection(command, stage, checking=True)
        self.alg.eval_mode()
        policy = self.alg.get_policy()
        with torch.inference_mode():
            while not command.check_finished:
                actions = policy(obs)
                obs, rewards, dones, _ = self.env.step(actions.to(self.env.device))
                if self.cfg.get("check_for_nan", True):
                    check_nan(obs, rewards, dones)
                obs = obs.to(self.device)
                policy.reset(dones.to(self.device))
                self._collect_results(command)
        self._write_curriculum_metrics(command)
        if self.gpu_global_rank == 0:
            result = "passed" if command.passed else "failed; starting training"
            print(
                f"[Curriculum] Stage {stage}: {result}; "
                f"mean relative velocity error={command.mean_error:.2%}"
            )
        return command.passed

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Check each stage first; otherwise use the latest training episodes."""
        command = self.env.unwrapped.command_manager.get_term("twist")
        if not isinstance(command, VelocityCurriculumCommand):
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        # Full episodes are required. Do not randomize episode_length_buf, which
        # would create shortened startup timeouts and invalidate the window.
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        it = start_it
        stage = 1
        needs_check = True

        while True:
            if needs_check:
                if self._check_stage(command, stage):
                    if stage == len(command.cfg.stages):
                        command.completed = True
                        break
                    stage += 1
                    continue
                obs = self._start_collection(command, stage, checking=False)
                self.alg.train_mode()
                needs_check = False
            if it >= total_it:
                break

            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self._collect_results(command)
                    extras.setdefault("log", {}).update(command.progress_metrics())
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"].get("rnd_cfg") else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                collect_time = time.perf_counter() - start
                if not command.passed:
                    self.alg.compute_returns(obs)

            start = time.perf_counter()
            if command.passed:
                # Keep the weights that met the rule. Do not apply another update
                # to the qualifying rollout or carry it into the next stage.
                self.alg.storage.clear()
                loss_dict = {}
            else:
                loss_dict = self.alg.update()
            learn_time = time.perf_counter() - start
            self.current_learning_iteration = it
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
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            it += 1
            if command.passed:
                if self.gpu_global_rank == 0:
                    print(f"[Curriculum] Stage {stage}: passed using the latest training episodes")
                if stage == len(command.cfg.stages):
                    command.completed = True
                    break
                stage += 1
                needs_check = True

        command.recording = False
        self._write_curriculum_metrics(command)
        if command.completed and self.gpu_global_rank == 0:
            print("[Curriculum] All velocity stages passed; training complete")
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
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
