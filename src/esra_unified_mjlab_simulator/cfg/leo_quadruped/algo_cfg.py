"""RL configuration for CCBR Leo robot velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass(kw_only=True)
class VelocityOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    start_stage: int = 1
    """First curriculum stage to train, using 1-based stage numbers."""


def ccbr_leo_ppo_runner_cfg() -> VelocityOnPolicyRunnerCfg:
    """Create RL runner configuration for CCBR Leo robot velocity task."""
    return VelocityOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="ccbr_leo_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
