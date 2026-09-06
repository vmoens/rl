# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run logging, RNG streams and evaluation of the DreamerV3 example."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase

from torchrl._utils import logger as torchrl_logger
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


# --- RNG streams -------------------------------------------------------------


POLICY_RNG_STREAM = 0
LEARNER_RNG_STREAM = 1
REPLAY_RNG_STREAM = 2


def stream_seed(seed: int, counter: int, stream: int) -> int:
    """Make one deterministic Torch seed from a seed, a counter and a stream.

    ``stream`` keeps its users independent: a change in the number of draws of
    one stream does not change the sequences of the others.
    """
    rng = np.random.default_rng(seed=[seed, counter, stream])
    words = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return (int(words[0]) << 32) | int(words[1])


# --- Run logging and episode bookkeeping -------------------------------------


def append_jsonl(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def latent_state_dim(cfg: DictConfig) -> int:
    return cfg.networks.num_categoricals * cfg.networks.num_classes


def collector_env_index(data: TensorDictBase, key: str = "env_index") -> torch.Tensor:
    """Return the environment index of each transition of an asynchronous batch."""
    env_index = data.get(key)
    if not isinstance(env_index, torch.Tensor):
        env_index = torch.as_tensor(env_index.tolist())
    return env_index.reshape(-1)


def split_by_env_index(
    data: TensorDictBase, env_index: torch.Tensor
) -> dict[int, TensorDictBase]:
    """Group the transitions of an asynchronous batch by environment.

    The transitions of one environment keep their arrival order, which is
    their time order.
    """
    # One densification and one stable sort, instead of one masked pass over
    # the lazy stack per environment.
    order = torch.argsort(env_index, stable=True)
    dense = data.to_tensordict()[order]
    indices, counts = torch.unique_consecutive(env_index[order], return_counts=True)
    return {
        int(index): chunk
        for index, chunk in zip(indices.tolist(), dense.split(counts.tolist(), 0))
    }


class AsyncEpisodeTracker:
    """Episode returns of environments whose transitions arrive in any order.

    Args:
        num_envs (int): Number of environments.
        milestone_key (str, optional): Key of a boolean vector of per-episode
            milestone flags under ``next``. Its value at the last step of an
            episode is reported with the episode.
    """

    def __init__(self, num_envs: int, milestone_key: str | None = None):
        self.returns = torch.zeros(num_envs)
        self.lengths = torch.zeros(num_envs, dtype=torch.long)
        self.milestone_key = milestone_key

    def update(
        self, data: TensorDictBase, env_index: torch.Tensor
    ) -> list[dict[str, object]]:
        """Consume a batch of transitions and return the episodes it completed."""
        reward = data.get(("next", "reward")).reshape(-1).cpu()
        done = data.get(("next", "done")).reshape(-1).cpu()
        milestones = (
            data.get(("next", self.milestone_key)).reshape(reward.numel(), -1).cpu()
            if self.milestone_key
            else None
        )
        completed = []
        for position in range(reward.numel()):
            env = int(env_index[position])
            self.returns[env] += float(reward[position])
            self.lengths[env] += 1
            if bool(done[position]):
                episode: dict[str, object] = {
                    "position": position,
                    "env_index": env,
                    "score": float(self.returns[env]),
                    "length": int(self.lengths[env]),
                }
                if milestones is not None:
                    episode["milestones"] = milestones[position].bool().tolist()
                completed.append(episode)
                self.returns[env] = 0
                self.lengths[env] = 0
        return completed


def training_episode_returns(
    data: TensorDictBase,
    running_return: torch.Tensor,
    num_envs: int,
) -> list[tuple[int, int, float]]:
    reward = data.get(("next", "reward")).squeeze(-1)
    done = data.get(("next", "done")).squeeze(-1)
    if num_envs == 1:
        reward = reward.reshape(1, -1)
        done = done.reshape(1, -1)
    completed = []
    for time_index in range(reward.shape[-1]):
        running_return.add_(reward[..., time_index].cpu())
        finished = done[..., time_index].cpu()
        completed.extend(
            (time_index, int(env_index), float(running_return[env_index]))
            for env_index in finished.nonzero().flatten()
        )
        running_return.masked_fill_(finished, 0)
    return completed


# --- Evaluation and plotting -------------------------------------------------


@torch.no_grad()
def eval_episode_reward(
    env: EnvBase,
    actor: TensorDictModuleBase,
    num_episodes: int,
    max_episode_steps: int,
) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(
                max_steps=max_episode_steps,
                policy=actor,
                break_when_any_done=True,
                auto_cast_to_device=True,
            )
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


def plot_enabled(cfg: DictConfig) -> bool:
    """Return True if the run must record the per-update losses."""
    return bool(cfg.logger.output_plot) and _has_matplotlib


def save_run_plot(
    cfg: DictConfig,
    eval_steps: list[int],
    eval_returns: list[torch.Tensor],
    loss_history: list[torch.Tensor],
    metric_names: tuple[str, ...],
) -> None:
    if not _has_matplotlib:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )
        return
    import matplotlib.pyplot as plt  # noqa: PLC0415

    returns = (
        (torch.stack(eval_returns) if eval_returns else torch.empty(0)).cpu().numpy()
    )
    losses = (
        torch.cat(loss_history) if loss_history else torch.empty(0, len(metric_names))
    ).numpy()
    column = {name: index for index, name in enumerate(metric_names)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(eval_steps, returns, marker="o")
    axes[0].set_title(f"{cfg.env.name} eval reward (real env)")
    axes[0].set_xlabel("env_step")
    axes[0].set_ylabel("avg episode return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses[:, column["loss_reconstruction"]], label="reco", alpha=0.8)
    axes[1].plot(losses[:, column["loss_reward"]], label="reward", alpha=0.8)
    axes[1].plot(
        losses[:, column["loss_dynamic"]] + losses[:, column["loss_representation"]],
        label="kl",
        alpha=0.8,
    )
    axes[1].set_title("World-model losses (update step)")
    axes[1].set_xlabel("update step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps"
    )
    fig.tight_layout()
    fig.savefig(cfg.logger.output_plot, dpi=120)
    torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
