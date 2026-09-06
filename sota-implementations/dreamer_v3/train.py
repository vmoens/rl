# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 training script that reproduces a pinned JAX configuration.

The script trains from flat vector observations, images, or both, and writes
its metrics to a JSONL file on the same step axis as the author-maintained JAX
implementation. Collection is synchronous (a ``Collector`` over a
``SerialEnv``) or asynchronous (an ``AsyncBatchedCollector``: environments step
independently while an inference server batches the policy calls).

Usage::

    python sota-implementations/dreamer_v3/train.py \\
        collector.total_frames=5000 logger.eval_every=500

    python sota-implementations/dreamer_v3/train.py \\
        --config-name=config_dmc_walker
"""
from __future__ import annotations

import copy
import importlib
from functools import partial
from pathlib import Path
from typing import NamedTuple

import hydra
import torch

from dreamer_v3_agent import (
    build_actor,
    build_continuation_model,
    build_imagination_model,
    build_mb_env,
    build_real_world_actor,
    build_value,
    build_world_model,
    DreamerV3BehaviorPolicySync,
    DreamerV3Optimizer,
    DreamerV3SeededPolicy,
    make_env,
    make_primed_env,
    pixels_key as configured_pixels_key,
    vector_key as configured_vector_key,
)
from dreamer_v3_replay import (
    collector_action_budget,
    DreamerV3ReplayPipeline,
    DreamerV3ReplayRecordBuilder,
    DreamerV3ReplaySampler,
    DreamerV3ShiftedRecordExtender,
    DreamerV3UpdateRatio,
    driver_step_for_action,
    MultiStreamReplay,
)
from dreamer_v3_utils import (
    append_jsonl,
    AsyncEpisodeTracker,
    collector_env_index,
    eval_episode_reward,
    latent_state_dim,
    LEARNER_RNG_STREAM,
    plot_enabled,
    REPLAY_RNG_STREAM,
    save_run_plot,
    split_by_env_index,
    stream_seed,
    training_episode_returns,
)
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import CudaGraphModule, TensorDictModuleBase

from torchrl import timeit
from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.collectors import AsyncBatchedCollector, Collector
from torchrl.data import LazyTensorStorage, OneHot, ReplayBuffer, RoundRobinWriter
from torchrl.envs import EnvBase, SerialEnv
from torchrl.envs.utils import ExplorationType
from torchrl.modules.inference_server import (
    InferenceDeviceConfig,
    InferenceServerConfig,
)
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
from torchrl.record.loggers import WandbLogger

# The per-update metrics, in the order of the tensor returned by the learner.
UPDATE_METRICS = (
    "loss_dynamic",
    "loss_representation",
    "loss_reconstruction",
    "loss_image_reconstruction",
    "loss_reward",
    "loss_continue",
    "loss_actor",
    "loss_value",
    "loss_replay_value",
    "grad_norm",
    "actor_entropy",
    "return_scale",
)
_METRIC_INDEX = {name: index for index, name in enumerate(UPDATE_METRICS)}


class ObservationLayout(NamedTuple):
    """The observation keys of the environment and their sizes."""

    vector_key: str | None
    vector_dim: int
    pixels_key: str | None
    pixels_shape: tuple[int, int, int] | None

    @property
    def observation_keys(self) -> tuple[tuple[str, str], ...]:
        keys = []
        if self.vector_key is not None:
            keys.append(("next", self.vector_key))
        if self.pixels_key is not None:
            keys.append(("next", self.pixels_key))
        return tuple(keys)


def observation_layout(cfg: DictConfig, env: EnvBase) -> ObservationLayout:
    """Read the configured observation keys and their shapes from the env."""
    vector = configured_vector_key(cfg)
    pixels = configured_pixels_key(cfg)
    if vector is None and pixels is None:
        raise ValueError("Set env.vector_key, env.pixels_key or both.")
    spec = env.observation_spec
    vector_dim = int(spec[vector].shape[-1]) if vector is not None else 0
    pixels_shape = (
        tuple(int(size) for size in spec[pixels].shape[-3:])
        if pixels is not None
        else None
    )
    return ObservationLayout(vector, vector_dim, pixels, pixels_shape)


class _Learner(NamedTuple):
    world_model: TensorDictModuleBase
    model_loss: DreamerV3ModelLoss
    actor_loss: DreamerV3ActorLoss
    value_loss: DreamerV3ValueLoss
    value_target_updater: SoftUpdate
    optimizer: DreamerV3Optimizer
    real_world_actor: TensorDictModuleBase


class _LearnerUpdate:
    """Run one complete DreamerV3 learner update."""

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        learner: _Learner,
        *,
        cudagraph_warmup: int = 5,
    ):
        self.cfg = cfg
        self.device = device
        self.model_loss = learner.model_loss
        self.actor_loss = learner.actor_loss
        self.value_loss = learner.value_loss
        self.optimizer = learner.optimizer
        self.value_target_updater = learner.value_target_updater
        self.state_dim = latent_state_dim(cfg)
        self.use_bfloat16 = cfg.optimization.mixed_precision and device.type == "cuda"
        self.pixels_key = configured_pixels_key(cfg)
        self.parameters = [
            parameter
            for group in learner.optimizer.param_groups
            for parameter in group["params"]
        ]
        train_step = self._forward_backward
        if cfg.optimization.cudagraph_train_step:
            if device.type != "cuda":
                raise RuntimeError(
                    "optimization.cudagraph_train_step requires a CUDA training device."
                )
            train_step = CudaGraphModule(
                train_step,
                warmup=cudagraph_warmup,
                device=device,
            )
        self.train_step = train_step

    def _forward_backward(
        self,
        sample: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16,
        ):
            model_loss_td, model_out = self.model_loss(sample)
            dynamic_loss = model_loss_td["loss_model_dynamic"]
            representation_loss = model_loss_td["loss_model_representation"]
            reco_loss = model_loss_td["loss_model_reco"]
            reward_loss = model_loss_td["loss_model_reward"]
            continue_loss = model_loss_td["loss_model_continue"]
            if self.pixels_key is not None:
                pixels = sample.get(("next", self.pixels_key))
                if not pixels.is_floating_point():
                    pixels = pixels.float() / 255.0
                reco_pixels = model_out.get(("next", "reco_pixels")).float()
                # The reference trains the image decoder with a squared error
                # summed over the pixels and averaged over batch and time.
                image_loss = (
                    (reco_pixels - pixels)
                    .pow(2)
                    .reshape(*sample.batch_size, -1)
                    .sum(-1)
                    .mean()
                    .reshape(1)
                )
            else:
                image_loss = torch.zeros_like(reco_loss)
            total_model_loss = (
                dynamic_loss
                + representation_loss
                + reco_loss
                + image_loss
                + reward_loss
                + continue_loss
            ).squeeze()

            post_state = (
                model_out.get(("next", "state")).detach().reshape(-1, self.state_dim)
            )
            post_belief = (
                model_out.get(("next", "belief"))
                .detach()
                .reshape(-1, cfg.networks.rnn_hidden_dim)
            )
            actor_input = TensorDict(
                {"state": post_state, "belief": post_belief},
                [post_state.shape[0]],
            )
            actor_loss_td, fake_data = self.actor_loss(actor_input)
            value_loss_td, _ = self.value_loss(fake_data.detach())

            replay_features = TensorDict(
                {
                    "state": model_out.get(("next", "state")),
                    "belief": model_out.get(("next", "belief")),
                    "bootstrap": fake_data.get("lambda_target")[..., 0, 0].reshape(
                        sample.batch_size
                    ),
                    "next": sample.get("next").select("reward", "done", "terminated"),
                },
                sample.batch_size,
            )
            replay_loss = self.value_loss.replay_value_loss(
                replay_features,
                horizon=cfg.optimization.continuation_horizon,
                lmbda=cfg.optimization.lmbda,
            )["loss_replay_value"]
            total_loss = (
                total_model_loss
                + actor_loss_td["loss_actor"]
                + value_loss_td["loss_value"]
                + cfg.optimization.replay_value_loss_weight * replay_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.parameters
            if parameter.grad is not None
        ]
        grad_norm = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(gradients))
        )
        metrics = torch.stack(
            (
                dynamic_loss.detach().reshape(()),
                representation_loss.detach().reshape(()),
                reco_loss.detach().reshape(()),
                image_loss.detach().reshape(()),
                reward_loss.detach().reshape(()),
                continue_loss.detach().reshape(()),
                actor_loss_td["loss_actor"].detach().reshape(()),
                value_loss_td["loss_value"].detach().reshape(()),
                replay_loss.detach().reshape(()),
                grad_norm.detach().reshape(()),
                actor_loss_td["actor_entropy"].detach().reshape(()),
                actor_loss_td["return_scale"].detach().reshape(()),
            )
        )
        return (
            metrics,
            model_out.get(("next", "state")).detach(),
            model_out.get(("next", "belief")).detach(),
        )

    def __call__(
        self,
        sample: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.train_step(sample)
        self.optimizer.step()
        self.value_target_updater.step()
        return result


def _validated_action_budget(cfg: DictConfig) -> int:
    num_envs = cfg.collector.num_envs
    if num_envs <= 0:
        raise ValueError(f"collector.num_envs must be positive, got {num_envs}.")
    if cfg.collector.total_frames < 0:
        if not cfg.optimization.max_time:
            raise ValueError(
                "collector.total_frames=-1 requires an optimization.max_time budget."
            )
        return -1
    if cfg.collector.backend == "sync" and cfg.collector.frames_per_batch % num_envs:
        raise ValueError(
            "collector.frames_per_batch must be divisible by collector.num_envs, "
            f"got {cfg.collector.frames_per_batch} and {num_envs}."
        )
    collector_action_frames = (
        collector_action_budget(
            cfg.collector.total_frames,
            num_envs,
            cfg.env.max_episode_steps,
        )
        if cfg.collector.count_reset_records
        else cfg.collector.total_frames
    )
    if collector_action_frames % cfg.collector.frames_per_batch:
        raise ValueError(
            "The action budget derived from collector.total_frames must be "
            "divisible by collector.frames_per_batch, got "
            f"{collector_action_frames} and {cfg.collector.frames_per_batch}."
        )
    return collector_action_frames


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _build_learner(
    cfg: DictConfig,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
    pixels_shape: tuple[int, int, int] | None = None,
    discrete: bool = False,
) -> _Learner:
    (
        world_model,
        prior_net,
        reward_net,
        reward_decoder,
        continuation_net,
    ) = build_world_model(
        cfg=cfg, obs_dim=obs_dim, action_dim=action_dim, pixels_shape=pixels_shape
    )
    world_model = world_model.to(device)
    imagination_model = build_imagination_model(
        prior_net=prior_net,
        reward_net=reward_net,
        reward_decoder=reward_decoder,
        # Compiling changes the categorical draws; "step" must match eager.
        compile_prior=cfg.optimization.compile_rssm == "scan",
    ).to(device)
    continuation_model = build_continuation_model(continuation_net=continuation_net).to(
        device
    )
    actor_model = build_actor(cfg=cfg, action_dim=action_dim, discrete=discrete).to(
        device
    )
    value_model = build_value(cfg=cfg).to(device)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        imagination_model=imagination_model,
        device=device,
    )

    vector = configured_vector_key(cfg) if obs_dim else None
    pixels = (
        (configured_pixels_key(cfg) or "pixels") if pixels_shape is not None else None
    )
    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_mode="separate",
        lambda_dynamic=cfg.optimization.dynamic_loss_weight,
        lambda_representation=cfg.optimization.representation_loss_weight,
        unimix=cfg.networks.unimix,
        # The vector reconstruction is a symlog squared error; the image
        # squared error is added by the learner update.
        lambda_reco=1.0 if vector is not None else 0.0,
        lambda_continue=1.0,
        continue_target_scale=1 - 1 / cfg.optimization.continuation_horizon,
        # The reference adds the event dimensions, then averages batch and time.
        global_average=False,
        detach_output=False,
    ).to(device)
    if vector is not None:
        model_loss.set_keys(
            pixels=vector,
            reco_pixels="reco_pixels" if pixels is None else f"reco_{vector}",
        )
    else:
        model_loss.set_keys(pixels=pixels)
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        continuation_model=continuation_model,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        return_normalization_rate=cfg.optimization.return_normalization_rate,
        return_normalization_min_scale=cfg.optimization.return_normalization_min_scale,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    actor_loss.to(device)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_critic_regularization=cfg.optimization.slow_critic_regularization,
    ).to(device)
    value_target_updater = SoftUpdate(value_loss, tau=cfg.optimization.slow_critic_tau)

    trainable_parameters = (
        list(world_model.parameters())
        + list(actor_model.parameters())
        + list(value_loss.parameters())
    )
    optimizer = DreamerV3Optimizer(
        trainable_parameters,
        lr=cfg.optimization.lr,
        agc=cfg.optimization.adaptive_grad_clip,
        eps=cfg.optimization.optimizer_eps,
        warmup_steps=cfg.optimization.warmup_steps,
    )

    real_world_actor = build_real_world_actor(
        world_model=world_model,
        actor_model=actor_model,
        mixed_precision=cfg.optimization.mixed_precision,
    )
    return _Learner(
        world_model=world_model,
        model_loss=model_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        value_target_updater=value_target_updater,
        optimizer=optimizer,
        real_world_actor=real_world_actor,
    )


def _build_collection(
    cfg: DictConfig,
    device: torch.device,
    learner: _Learner,
    state_dim: int,
    action_dim: int,
    collector_action_frames: int,
) -> tuple[Collector | AsyncBatchedCollector, DreamerV3BehaviorPolicySync | None]:
    num_envs = cfg.collector.num_envs
    real_world_actor = learner.real_world_actor
    async_collection = cfg.collector.backend == "async"
    if async_collection and not cfg.optimization.deferred_policy_sync:
        raise ValueError(
            "collector.backend='async' requires optimization.deferred_policy_sync="
            "true: the inference server owns its own copy of the policy."
        )
    if cfg.optimization.deferred_policy_sync:
        collector_actor = copy.deepcopy(real_world_actor)
        # The decoder cannot act, but the reference syncs both parameter trees.
        behavior_decoder = copy.deepcopy(learner.world_model[2])
        learner_policy_tree = torch.nn.ModuleList(
            [real_world_actor, learner.world_model[2]]
        )
        behavior_policy_tree = torch.nn.ModuleList([collector_actor, behavior_decoder])
        behavior_policy_sync = DreamerV3BehaviorPolicySync(
            learner_policy_tree, behavior_policy_tree
        )
    else:
        collector_actor = real_world_actor
        behavior_policy_sync = None
    collector_policy = (
        DreamerV3SeededPolicy(collector_actor, cfg.env.seed)
        if cfg.optimization.separate_policy_rng
        else collector_actor
    )

    def explore_seed(index: int) -> int | None:
        return cfg.env.seed + 2 + index if cfg.env.use_seed else None

    if async_collection:
        policy_device = (
            torch.device(cfg.collector.policy_device)
            if cfg.collector.policy_device
            else device
        )
        collector_actor.to(policy_device)
        served_policy = collector_policy
        if cfg.collector.compile_policy:
            served_policy = torch.compile(collector_policy, dynamic=True)
        cpu = torch.device("cpu")
        collector = AsyncBatchedCollector(
            create_env_fn=[
                partial(
                    make_primed_env,
                    cfg,
                    explore_seed(index),
                    state_dim,
                    action_dim,
                    env_index=index,
                )
                for index in range(num_envs)
            ],
            policy=served_policy,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=collector_action_frames,
            env_backend=cfg.collector.env_backend,
            env_exchange=cfg.collector.get("env_exchange", "queue"),
            policy_backend="threading",
            device_config=InferenceDeviceConfig(
                policy_device=policy_device,
                output_device=cpu,
                env_device=cpu,
                storing_device=cpu,
            ),
            server_config=InferenceServerConfig(
                max_batch_size=cfg.collector.inference_max_batch_size,
                min_batch_size=cfg.collector.inference_min_batch_size,
                timeout=cfg.collector.inference_timeout,
            ),
            result_queue_maxsize=cfg.collector.max_pending_frames,
        )
    else:
        if num_envs == 1:
            explore_env = make_primed_env(
                cfg, explore_seed(0), state_dim, action_dim, env_index=0
            )
        else:
            explore_env = SerialEnv(
                num_envs,
                [
                    partial(
                        make_primed_env,
                        cfg,
                        explore_seed(index),
                        state_dim,
                        action_dim,
                        env_index=index,
                    )
                    for index in range(num_envs)
                ],
            )
        collector = Collector(
            explore_env,
            collector_policy,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=collector_action_frames,
            policy_device=device,
            env_device="cpu",
            storing_device="cpu",
            exploration_type=ExplorationType.RANDOM
            if cfg.collector.exploration == "random"
            else ExplorationType.MODE,
        )
    if cfg.optimization.separate_policy_rng:
        # The collector's construction-time policy call is not an action.
        collector_policy.reset_counter()
    return collector, behavior_policy_sync


def _apply_behavior_sync(
    collector: Collector | AsyncBatchedCollector,
    behavior_policy_sync: DreamerV3BehaviorPolicySync,
) -> None:
    """Install the staged learner parameters into the behavior policy."""
    if not behavior_policy_sync.has_pending:
        return
    server = getattr(collector, "_server", None)
    if server is not None:
        # The inference server may be running a forward pass on another thread.
        server.update_model(lambda _: behavior_policy_sync.apply_after_action())
    else:
        behavior_policy_sync.apply_after_action()


def _build_replay(
    cfg: DictConfig,
    num_envs: int,
    replay_device: torch.device,
    observation_keys: tuple[tuple[str, str], ...],
) -> tuple[
    ReplayBuffer | MultiStreamReplay,
    DreamerV3ReplaySampler | None,
    DreamerV3ReplayRecordBuilder | None,
    DreamerV3ShiftedRecordExtender | None,
    DreamerV3ReplayPipeline,
]:
    replay_pipeline = DreamerV3ReplayPipeline()
    if cfg.collector.backend == "async":
        replay = MultiStreamReplay(
            num_envs,
            buffer_size=cfg.replay_buffer.buffer_size,
            slice_len=cfg.replay_buffer.seq_len + 1,
            num_sequences=cfg.replay_buffer.batch_size,
            online=cfg.replay_buffer.online,
            seed=stream_seed(cfg.env.seed, 0, REPLAY_RNG_STREAM) % 2**62,
            device=replay_device,
            observation_keys=observation_keys,
        )
        return replay, None, None, None, replay_pipeline
    replay_sampler = DreamerV3ReplaySampler(
        # The extra record receives the last refreshed posterior.
        slice_len=cfg.replay_buffer.seq_len + 1,
        online=cfg.replay_buffer.online,
    )
    rb = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.replay_buffer.buffer_size,
            ndim=2 if num_envs > 1 else 1,
            device=replay_device,
        ),
        dim_extend=1 if num_envs > 1 else 0,
        writer=RoundRobinWriter(track_generations=True),
        sampler=replay_sampler,
        batch_size=cfg.replay_buffer.batch_size * (cfg.replay_buffer.seq_len + 1),
        generator=torch.Generator().manual_seed(
            stream_seed(cfg.env.seed, 0, REPLAY_RNG_STREAM)
        ),
    )
    replay_record_builder = DreamerV3ReplayRecordBuilder(num_envs, observation_keys)
    shifted_record_extender = (
        DreamerV3ShiftedRecordExtender(num_envs)
        if cfg.collector.count_reset_records
        else None
    )
    return (
        rb,
        replay_sampler,
        replay_record_builder,
        shifted_record_extender,
        replay_pipeline,
    )


class _RunLogger:
    """Append run records to JSONL and mirror their scalars to Weights & Biases."""

    def __init__(self, cfg: DictConfig, jsonl_path: Path | None):
        self.jsonl_path = jsonl_path
        self.milestone_names = list(cfg.env.get("milestone_names", None) or [])
        self.wandb = None
        if cfg.logger.get("backend", None) == "wandb":
            wandb_kwargs = {
                "entity": cfg.logger.entity,
                "group": cfg.logger.group,
                "tags": list(cfg.logger.tags) or None,
                "mode": cfg.logger.mode,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.logger.get("base_url", None):
                # An explicit server wins over WANDB_BASE_URL, which imported
                # libraries may have redirected to their own instance.
                wandb = importlib.import_module("wandb")
                wandb_kwargs["settings"] = wandb.Settings(base_url=cfg.logger.base_url)
            self.wandb = WandbLogger(
                exp_name=cfg.logger.exp_name or f"dreamer_v3_{cfg.env.name}",
                project=cfg.logger.project,
                **{
                    key: value
                    for key, value in wandb_kwargs.items()
                    if value is not None
                },
            )
            # Every curve is plotted against environment steps.
            self.wandb.experiment.define_metric("environment_steps")
            self.wandb.experiment.define_metric("*", step_metric="environment_steps")

    def log(self, record: dict[str, object]) -> None:
        append_jsonl(self.jsonl_path, record)
        if self.wandb is None:
            return
        kind = record.get("type", "run")
        payload: dict[str, object] = {}
        for key, value in record.items():
            if key in ("type", "environment_steps"):
                continue
            if key == "milestones":
                for name, flag in zip(self.milestone_names, value):
                    payload[f"{kind}/obtained_{name}"] = float(flag)
            elif isinstance(value, bool | int | float):
                payload[f"{kind}/{key}"] = value
        if "environment_steps" in record:
            payload["environment_steps"] = record["environment_steps"]
        self.wandb.experiment.log(payload)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.experiment.finish()


class _ThroughputWindow:
    """Collection and learning rates since the previous training log."""

    def __init__(self, run_timer, records_per_update: int):
        self.run_timer = run_timer
        self.records_per_update = records_per_update
        self.mark(0, 0, 0)

    def mark(self, record_step: int, action_step: int, update_step: int) -> None:
        self.time = self.run_timer.elapsed()
        self.record_step = record_step
        self.action_step = action_step
        self.update_step = update_step

    def rates(
        self, record_step: int, action_step: int, update_step: int
    ) -> dict[str, float]:
        elapsed = max(self.run_timer.elapsed() - self.time, 1e-9)
        records = record_step - self.record_step
        updates = update_step - self.update_step
        return {
            "records_per_second": records / elapsed,
            "actions_per_second": (action_step - self.action_step) / elapsed,
            "updates_per_second": updates / elapsed,
            "replayed_steps_per_record": updates
            * self.records_per_update
            / max(records, 1),
        }


def _log_train_episodes(
    run_logger: _RunLogger,
    cfg: DictConfig,
    completed_episodes: list[tuple[int, int, float]],
    batch_start_action_step: int,
) -> None:
    num_envs = cfg.collector.num_envs
    for time_index, env_index, score in completed_episodes:
        if cfg.collector.count_reset_records:
            action_index = batch_start_action_step // num_envs + time_index + 1
            episode_step = driver_step_for_action(
                action_index,
                env_index,
                num_envs,
                cfg.env.max_episode_steps,
            )
        else:
            episode_step = (
                batch_start_action_step + time_index * num_envs + env_index + 1
            )
        run_logger.log(
            {
                "type": "train_episode",
                "environment_steps": episode_step,
                "score": score,
            }
        )


def _log_async_episodes(
    run_logger: _RunLogger,
    episodes: list[dict[str, object]],
    batch_start_action_step: int,
) -> None:
    for episode in episodes:
        record: dict[str, object] = {
            "type": "train_episode",
            "environment_steps": batch_start_action_step + episode["position"] + 1,
            "score": episode["score"],
            "length": episode["length"],
            "env_index": episode["env_index"],
        }
        if "milestones" in episode:
            record["milestones"] = episode["milestones"]
        run_logger.log(record)


def _log_train_window(
    *,
    run_logger: _RunLogger,
    run_timer,
    record_step: int,
    action_step: int,
    update_step: int,
    loss_window_sum: torch.Tensor,
    loss_window_updates: int,
    throughput: dict[str, float],
    replay_stats: dict[str, float],
    server_stats: dict[str, float],
) -> None:
    metrics = (loss_window_sum / max(loss_window_updates, 1)).cpu().tolist()
    timings = {
        f"time_{name.replace('/', '_')}": value
        for name, value in timeit.todict(percall=True).items()
    }
    run_logger.log(
        {
            "type": "train",
            "environment_steps": record_step,
            "action_steps": action_step,
            "updates": update_step,
            "updates_in_window": loss_window_updates,
            **dict(zip(UPDATE_METRICS, metrics)),
            **throughput,
            **{f"replay_{key}": value for key, value in replay_stats.items()},
            **{f"inference_{key}": value for key, value in server_stats.items()},
            **timings,
            "elapsed_seconds": run_timer.elapsed(),
        }
    )


def _evaluate(
    *,
    cfg: DictConfig,
    device: torch.device,
    eval_env,
    real_world_actor: TensorDictModuleBase,
    run_logger: _RunLogger,
    run_timer,
    record_step: int,
    latest_losses: torch.Tensor,
) -> torch.Tensor:
    # Evaluation samples RSSM latents, thus a fork keeps training unchanged.
    with timeit("dreamer_v3/evaluation"), torch.random.fork_rng(
        devices=[device] if device.type == "cuda" else []
    ):
        r = eval_episode_reward(
            eval_env,
            real_world_actor,
            cfg.logger.eval_episodes,
            cfg.env.max_episode_steps,
        )
    torchrl_logger.info(
        "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
        record_step,
        r.item(),
        (
            latest_losses[_METRIC_INDEX["loss_dynamic"]]
            + latest_losses[_METRIC_INDEX["loss_representation"]]
        ).item(),
        latest_losses[_METRIC_INDEX["loss_reconstruction"]].item(),
        latest_losses[_METRIC_INDEX["loss_reward"]].item(),
        latest_losses[_METRIC_INDEX["loss_actor"]].item(),
    )
    run_logger.log(
        {
            "type": "evaluation",
            "environment_steps": record_step,
            "return": r.item(),
            "episodes": cfg.logger.eval_episodes,
            "elapsed_seconds": run_timer.elapsed(),
        }
    )
    return r


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)

    device = (
        torch.device(cfg.optimization.device)
        if cfg.optimization.device
        else get_available_device()
    )
    replay_device = (
        torch.device(cfg.replay_buffer.device) if cfg.replay_buffer.device else device
    )
    use_bfloat16 = cfg.optimization.mixed_precision and device.type == "cuda"
    async_collection = cfg.collector.backend == "async"
    torchrl_logger.info(
        "DreamerV3 execution: device=%s, replay_device=%s, collector=%s, "
        "rssm_backend=%s, rssm_scan_unroll=%s, mixed_precision=%s, "
        "cudagraph_train_step=%s",
        device,
        replay_device,
        cfg.collector.backend,
        cfg.optimization.compile_rssm or "eager",
        (
            cfg.optimization.rssm_scan_unroll
            if cfg.optimization.compile_rssm == "scan"
            else "n/a"
        ),
        use_bfloat16,
        cfg.optimization.cudagraph_train_step,
    )
    num_envs = cfg.collector.num_envs
    count_reset_records = cfg.collector.count_reset_records
    if async_collection and count_reset_records:
        raise ValueError(
            "collector.count_reset_records is only supported by the synchronous "
            "collector."
        )
    collector_action_frames = _validated_action_budget(cfg)
    real_env = make_env(cfg, cfg.env.seed)
    layout = observation_layout(cfg, real_env)
    discrete = isinstance(real_env.action_spec, OneHot)
    action_dim = real_env.action_spec.shape[-1]
    state_dim = latent_state_dim(cfg)
    metrics_jsonl_path = (
        Path(cfg.logger.metrics_jsonl).resolve() if cfg.logger.metrics_jsonl else None
    )
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("")
    run_logger = _RunLogger(cfg, metrics_jsonl_path)
    timeit.reset()
    run_timer = timeit("dreamer_v3/run").start()

    learner = _build_learner(
        cfg,
        device,
        layout.vector_dim,
        action_dim,
        pixels_shape=layout.pixels_shape,
        discrete=discrete,
    )
    parameter_counts = {
        "world_model": _count_parameters(learner.world_model),
        "actor": _count_parameters(learner.actor_loss.actor_model),
        "value": _count_parameters(learner.value_loss.value_model),
    }
    torchrl_logger.info(
        "DreamerV3 parameters: world model %d, actor %d, value %d (%s actions, "
        "vector %s, pixels %s)",
        parameter_counts["world_model"],
        parameter_counts["actor"],
        parameter_counts["value"],
        "discrete" if discrete else "continuous",
        layout.vector_dim,
        layout.pixels_shape,
    )
    learner_update = _LearnerUpdate(cfg, device, learner)

    collector, behavior_policy_sync = _build_collection(
        cfg, device, learner, state_dim, action_dim, collector_action_frames
    )
    (
        replay,
        replay_sampler,
        replay_record_builder,
        shifted_record_extender,
        replay_pipeline,
    ) = _build_replay(cfg, num_envs, replay_device, layout.observation_keys)

    action_step = 0
    # Each worker sends a reset record before its first control transition.
    record_step = num_envs if count_reset_records else 0
    update_step = 0
    running_training_return = torch.zeros(num_envs)
    episode_tracker = (
        AsyncEpisodeTracker(num_envs, cfg.env.get("milestone_key", None))
        if async_collection
        else None
    )
    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_history: list[torch.Tensor] = []
    loss_window_sum = torch.zeros(len(UPDATE_METRICS), device=device)
    loss_window_updates = 0
    record_loss_history = plot_enabled(cfg)
    next_eval = 0
    next_train_log = 0

    eval_env = (
        make_primed_env(cfg, cfg.env.seed + 100, state_dim, action_dim)
        if cfg.logger.eval_every
        else None
    )

    sequence_records = cfg.replay_buffer.seq_len + 1
    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )
    if not async_collection:
        warmup = max(warmup, num_envs * sequence_records)

    updates_per_batch = cfg.optimization.updates_per_batch
    records_per_update = cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len
    update_ratio = (
        DreamerV3UpdateRatio(cfg.optimization.train_ratio / records_per_update)
        if cfg.optimization.train_ratio is not None
        else None
    )
    throughput = _ThroughputWindow(run_timer, records_per_update)
    max_time = cfg.optimization.max_time
    stop_reason = "frame_budget"

    if cfg.optimization.separate_policy_rng:
        # Keep the learner draws in a range apart from the policy stream.
        torch.manual_seed(stream_seed(cfg.env.seed, 0, LEARNER_RNG_STREAM))

    for data in collector:
        if max_time and run_timer.elapsed() >= max_time:
            stop_reason = "max_time"
            break
        # The next action is already computed, thus it keeps the older policy.
        if behavior_policy_sync is not None:
            _apply_behavior_sync(collector, behavior_policy_sync)
        batch_start_action_step = action_step
        if async_collection:
            env_index = collector_env_index(data)
            _log_async_episodes(
                run_logger,
                episode_tracker.update(data, env_index),
                batch_start_action_step,
            )
            with timeit("dreamer_v3/replay_extend"):
                for stream, stream_data in split_by_env_index(data, env_index).items():
                    record_step += replay.extend_stream(stream, stream_data)
            action_step += data.numel()
            replay_ready = replay.num_sampleable_streams > 0
        else:
            completed_episodes = training_episode_returns(
                data, running_training_return, num_envs
            )
            _log_train_episodes(
                run_logger, cfg, completed_episodes, batch_start_action_step
            )
            replay_data = replay_record_builder(data)
            with timeit("dreamer_v3/replay_extend"):
                if shifted_record_extender is not None:
                    shifted_record_extender.extend(replay, replay_sampler, replay_data)
                else:
                    replay_indices = replay.extend(
                        replay_data if num_envs > 1 else replay_data.reshape(-1)
                    )
                    replay_sampler.observe_extend(replay_indices, replay.storage)
            action_step += data.numel()
            record_step += replay_data.numel() if count_reset_records else data.numel()
            replay_ready = len(replay) >= num_envs * sequence_records
        if not replay_pipeline.has_prefetched and update_step == 0 and replay_ready:
            # Prefetch one batch ahead of the learner warmup gate below.
            with timeit("dreamer_v3/replay_sample"):
                replay_pipeline.prefetch(replay)

        if len(replay) < warmup or not replay_ready:
            continue

        batch_updates = (
            update_ratio(record_step) if update_ratio is not None else updates_per_batch
        )
        if not batch_updates:
            continue

        if behavior_policy_sync is not None:
            # Stage one time per batch; more updates keep the pending snapshot.
            behavior_policy_sync.stage_before_training()

        batch_losses = torch.empty((batch_updates, len(UPDATE_METRICS)), device=device)
        for update_index in range(batch_updates):
            with timeit("dreamer_v3/replay_sample"):
                replay_sample, sample_info = replay_pipeline.take(replay)
                replay_sample = replay_sample.reshape(
                    cfg.replay_buffer.batch_size,
                    sequence_records,
                )
                sample = replay_sample[:, :-1].to(device)
            with timeit("dreamer_v3/replay_update"):
                # Apply the older refresh, thus replay stays one sample ahead.
                replay_pipeline.apply_pending_context(replay)
            with timeit("dreamer_v3/train_update"):
                (
                    update_losses,
                    refreshed_state,
                    refreshed_belief,
                ) = learner_update(sample)
                batch_losses[update_index].copy_(update_losses)
                loss_window_sum += update_losses
                loss_window_updates += 1
            with timeit("dreamer_v3/replay_update"):
                replay_pipeline.stage_context(
                    sample_info,
                    refreshed_state,
                    refreshed_belief,
                )
            update_step += 1

        if record_loss_history:
            loss_history.append(batch_losses.cpu())

        train_log_due = bool(
            cfg.logger.train_every
            and (
                record_step >= next_train_log
                or (
                    collector_action_frames > 0
                    and action_step >= collector_action_frames
                )
            )
        )
        eval_due = bool(
            eval_env is not None and cfg.logger.eval_every and record_step >= next_eval
        )
        latest_losses = batch_losses[-1].cpu() if eval_due else None
        if train_log_due:
            replay_stats = {"records": len(replay)}
            if async_collection:
                replay_stats["sampleable_streams"] = replay.num_sampleable_streams
            _log_train_window(
                run_logger=run_logger,
                run_timer=run_timer,
                record_step=record_step,
                action_step=action_step,
                update_step=update_step,
                loss_window_sum=loss_window_sum,
                loss_window_updates=loss_window_updates,
                throughput=throughput.rates(record_step, action_step, update_step),
                replay_stats=replay_stats,
                server_stats=(
                    collector.server_stats(reset=True) if async_collection else {}
                ),
            )
            throughput.mark(record_step, action_step, update_step)
            loss_window_sum.zero_()
            loss_window_updates = 0
            next_train_log = record_step + cfg.logger.train_every

        if eval_due:
            r = _evaluate(
                cfg=cfg,
                device=device,
                eval_env=eval_env,
                real_world_actor=learner.real_world_actor,
                run_logger=run_logger,
                run_timer=run_timer,
                record_step=record_step,
                latest_losses=latest_losses,
            )
            history_steps.append(record_step)
            history_eval.append(r)
            next_eval = record_step + cfg.logger.eval_every

    collector.shutdown()
    if cfg.logger.output_plot:
        save_run_plot(cfg, history_steps, history_eval, loss_history, UPDATE_METRICS)

    run_logger.log(
        {
            "type": "summary",
            "backend": cfg.env.backend,
            "collector": cfg.collector.backend,
            "environment": cfg.env.name,
            "task": cfg.env.task,
            "seed": cfg.env.seed,
            "environment_seeded": bool(cfg.env.use_seed),
            "stop_reason": stop_reason,
            "total_environment_steps": record_step,
            "total_action_steps": action_step,
            "updates": update_step,
            "bfloat16": use_bfloat16,
            "elapsed_seconds": run_timer.elapsed(),
            **{f"parameters_{name}": count for name, count in parameter_counts.items()},
            "timings": timeit.todict(percall=False),
        }
    )
    run_logger.finish()
    if metrics_jsonl_path is not None:
        torchrl_logger.info("Saved run metrics to %s", metrics_jsonl_path)


if __name__ == "__main__":
    main()
