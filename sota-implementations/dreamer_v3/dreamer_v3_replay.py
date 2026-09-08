# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Driver-step accounting and the continuous replay stream of the example."""
from __future__ import annotations

from typing import TypeAlias

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.utils import unravel_key

from torchrl.data import LazyTensorStorage, ReplayBuffer, RoundRobinWriter, SliceSampler

ReplayIndex: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
ReplaySampleInfo: TypeAlias = dict[str, ReplayIndex]
_REPLAY_CONTEXT_VALID_KEY = ("collector", "context_valid")
_DEFAULT_OBSERVATION_KEYS: tuple[NestedKey, ...] = (("next", "observation"),)


# --- Driver step accounting --------------------------------------------------


def driver_step_for_action(
    action_index: int,
    env_index: int,
    num_envs: int,
    max_episode_steps: int,
) -> int:
    """Return the driver step of a one-based action index, with reset records."""
    reset_records = 1 + (action_index - 1) // max_episode_steps
    vector_record = action_index + reset_records
    return (vector_record - 1) * num_envs + env_index + 1


def collector_action_budget(
    record_budget: int,
    num_envs: int,
    max_episode_steps: int,
) -> int:
    """Return the actions in a driver-record budget that also holds resets."""
    if record_budget % num_envs:
        raise ValueError(
            "A driver-record budget must be divisible by the number of "
            f"environments, got {record_budget} and {num_envs}."
        )
    vector_records = record_budget // num_envs
    reset_records = (vector_records + max_episode_steps) // (max_episode_steps + 1)
    return (vector_records - reset_records) * num_envs


class DreamerV3UpdateRatio:
    """Schedule learner updates from a ratio of updates to driver records.

    Each call truncates the count from the cumulative driver-record count and
    keeps the remainder. The first call returns one update.

    Args:
        ratio (float): Learner updates for each driver record.
    """

    def __init__(self, ratio: float):
        self.ratio = ratio
        self._previous: float | None = None

    def __call__(self, record_count: int) -> int:
        if self.ratio <= 0:
            return 0
        if self._previous is None:
            self._previous = float(record_count)
            return 1
        repeats = int((record_count - self._previous) * self.ratio)
        self._previous += repeats / self.ratio
        return repeats


# --- Replay: record stream, writeback, sampling ------------------------------


def _refresh_replay_context(
    replay_buffer: ReplayBuffer,
    sample_indices: ReplayIndex,
    sample_generations: torch.Tensor,
    state: torch.Tensor,
    belief: torch.Tensor,
) -> None:
    if not isinstance(sample_indices, tuple):
        sample_indices = (sample_indices,)
    batch_size, sequence_length = state.shape[:2]
    context_length = sequence_length + 1
    destination_indices = tuple(
        index.reshape(batch_size, context_length)[:, 1:].reshape(-1)
        for index in sample_indices
    )
    destination_generation = sample_generations.reshape(batch_size, context_length)[
        :, 1:
    ].reshape(-1)
    # Slices overlap, and a CUDA index write leaves duplicate coordinates
    # undefined. Keep the last value of each coordinate.
    coordinates = torch.stack(destination_indices, -1)
    linear_coordinate = coordinates[:, 0]
    for coordinate, size in zip(
        coordinates[:, 1:].unbind(-1), replay_buffer.storage.shape[1:]
    ):
        linear_coordinate = linear_coordinate * int(size) + coordinate
    order = linear_coordinate.argsort(stable=True)
    ordered_coordinate = linear_coordinate[order]
    keep_ordered = torch.ones_like(ordered_coordinate, dtype=torch.bool)
    keep_ordered[:-1] = ordered_coordinate[:-1] != ordered_coordinate[1:]
    keep = order[keep_ordered]
    destination_indices = tuple(index[keep] for index in destination_indices)
    destination_index = (
        torch.stack(destination_indices, -1)
        if len(destination_indices) > 1
        else destination_indices[0]
    )
    destination_generation = destination_generation[keep]
    replay_buffer.update_if_present(
        index=destination_index,
        generation=destination_generation,
        patch={
            "state": state.detach()
            .float()
            .reshape(-1, state.shape[-1])[keep.to(state.device)],
            "belief": belief.detach()
            .float()
            .reshape(-1, belief.shape[-1])[keep.to(belief.device)],
            _REPLAY_CONTEXT_VALID_KEY: torch.ones(
                (keep.numel(), 1), dtype=torch.bool, device=state.device
            ),
        },
    )


class DreamerV3ReplayPipeline:
    """Sample one batch ahead, and apply each latent refresh one update behind."""

    def __init__(self):
        self._prefetched: tuple[TensorDictBase, ReplaySampleInfo] | None = None
        self._pending_context: tuple[
            ReplaySampleInfo, torch.Tensor, torch.Tensor
        ] | None = None

    @property
    def has_prefetched(self) -> bool:
        return self._prefetched is not None

    @property
    def has_pending_context(self) -> bool:
        return self._pending_context is not None

    def prefetch(self, replay_buffer: ReplayBuffer) -> None:
        if self._prefetched is None:
            self._prefetched = replay_buffer.sample(return_info=True)

    def take(
        self, replay_buffer: ReplayBuffer
    ) -> tuple[TensorDictBase, ReplaySampleInfo]:
        """Return the prefetched batch, and sample the next one."""
        self.prefetch(replay_buffer)
        current = self._prefetched
        self._prefetched = replay_buffer.sample(return_info=True)
        return current

    def apply_pending_context(
        self, replay_buffer: ReplayBuffer | MultiStreamReplay
    ) -> None:
        """Apply the previous refresh, after ``take`` samples the next batch."""
        if self._pending_context is not None:
            pending_info, pending_state, pending_belief = self._pending_context
            if isinstance(replay_buffer, MultiStreamReplay):
                replay_buffer.refresh_context(
                    pending_info, pending_state, pending_belief
                )
            else:
                _refresh_replay_context(
                    replay_buffer,
                    pending_info["index"],
                    pending_info["index_generation"],
                    pending_state,
                    pending_belief,
                )
            self._pending_context = None

    def stage_context(
        self,
        sample_info: ReplaySampleInfo,
        state: torch.Tensor,
        belief: torch.Tensor,
    ) -> None:
        if self._pending_context is not None:
            raise RuntimeError(
                "The preceding replay context must be applied before staging "
                "another learner output."
            )
        self._pending_context = (sample_info, state, belief)


class DreamerV3ReplayRecordBuilder:
    """Convert collector transitions into the replay stream.

    Args:
        num_streams (int): Number of environment streams in the collector data.
        observation_keys (tuple of NestedKey, optional): The ``("next", ...)``
            observation entries copied into each record. Defaults to
            ``(("next", "observation"),)``.
    """

    def __init__(
        self,
        num_streams: int,
        observation_keys: tuple[NestedKey, ...] = _DEFAULT_OBSERVATION_KEYS,
    ):
        self.num_streams = num_streams
        self.observation_keys = tuple(unravel_key(key) for key in observation_keys)
        for key in self.observation_keys:
            if not isinstance(key, tuple) or key[0] != "next" or len(key) < 2:
                raise ValueError(
                    "Replay observation keys must be nested under 'next', got "
                    f"{key!r}."
                )
        self._started = False

    @staticmethod
    def _root_key(key: tuple[str, ...]) -> NestedKey:
        return key[1] if len(key) == 2 else key[1:]

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        if self.num_streams == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2 or data.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected collector data with shape [num_streams, time], got "
                f"{tuple(data.shape)} for {self.num_streams} streams."
            )

        record_keys = (
            "action",
            "is_init",
            "state",
            "belief",
            *self.observation_keys,
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
        )
        num_steps = data.shape[1]
        if not num_steps:
            return data.select(*record_keys, strict=True).clone()
        reset = data.get("is_init").reshape(self.num_streams, num_steps, -1).any(-1)
        insert_reset = reset.any(0)
        unsynchronized = insert_reset & ~reset.all(0)
        if not self._started:
            # The first record of the stream needs no reset record before it.
            insert_reset = insert_reset.clone()
            insert_reset[0] = False
            unsynchronized = unsynchronized.clone()
            unsynchronized[0] = False
        if unsynchronized.any():
            raise RuntimeError(
                "The 2D DreamerV3 replay stream requires synchronized episode "
                "resets across collector environments."
            )
        self._started = True

        transition = data.select(*record_keys, strict=True).clone()
        # These records model the transitions into next.observation, so they
        # keep their action; separate reset records mark the resets.
        transition.get("is_init").zero_()
        transition.set(
            _REPLAY_CONTEXT_VALID_KEY,
            torch.ones_like(transition.get("is_init"), dtype=torch.bool),
        )
        reset_steps = insert_reset.nonzero().squeeze(-1)
        num_resets = reset_steps.numel()
        if not num_resets:
            return transition

        reset_records = transition[:, reset_steps].clone()
        for key in (
            "action",
            "state",
            "belief",
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
        ):
            reset_records.get(key).zero_()
        reset_records.get("is_init").fill_(True)
        reset_sources = data[:, reset_steps]
        for key in self.observation_keys:
            reset_records.set(key, reset_sources.get(self._root_key(key)).clone())

        # Each reset record precedes the transition whose root observation it
        # carries: transition t lands after the resets at or before t.
        transition_positions = torch.arange(num_steps) + insert_reset.cumsum(0)
        reset_positions = transition_positions[reset_steps] - 1
        order = torch.empty(num_steps + num_resets, dtype=torch.long)
        order[transition_positions] = torch.arange(num_steps)
        order[reset_positions] = num_steps + torch.arange(num_resets)
        return torch.cat([transition, reset_records], 1)[:, order]


class DreamerV3ShiftedRecordExtender:
    """Keep a placeholder record for the posterior of the newest transition.

    The next collector batch completes it in place, at the same slot.
    """

    def __init__(self, num_streams: int):
        self.num_streams = num_streams
        self._tail_index: torch.Tensor | None = None
        self._tail_generation: torch.Tensor | None = None

    @staticmethod
    def _tail_placeholder(records: TensorDictBase) -> TensorDictBase:
        tail = records[:, -1].clone()
        tail.get("action").zero_()
        tail.get("is_init").zero_()
        tail.get("state").zero_()
        tail.get("belief").zero_()
        tail.get(("next", "reward")).zero_()
        tail.get(("next", "done")).zero_()
        tail.get(("next", "terminated")).zero_()
        tail.get(_REPLAY_CONTEXT_VALID_KEY).zero_()
        return tail.unsqueeze(1)

    def _finalize_tail(
        self, replay_buffer: ReplayBuffer, records: TensorDictBase
    ) -> None:
        tail_index = self._tail_index
        tail_generation = self._tail_generation
        if tail_index is None or tail_generation is None:
            return

        storage = replay_buffer.storage
        stored = (
            storage[tail_index]
            if storage.ndim == 1
            else storage[tuple(tail_index.unbind(-1))]
        )
        incoming = records[:, 0].clone().to(stored.device)
        context_valid = stored.get(_REPLAY_CONTEXT_VALID_KEY)
        incoming.set(
            "state",
            torch.where(context_valid, stored.get("state"), incoming.get("state")),
        )
        incoming.set(
            "belief",
            torch.where(context_valid, stored.get("belief"), incoming.get("belief")),
        )
        incoming.set(
            _REPLAY_CONTEXT_VALID_KEY,
            torch.ones_like(context_valid, dtype=torch.bool),
        )
        result = replay_buffer.update_if_present(
            index=tail_index,
            generation=tail_generation,
            patch=incoming,
        )
        if result.updated_count != self.num_streams:
            raise RuntimeError(
                "The mutable DreamerV3 replay tail was recycled before it "
                "could be finalized."
            )

    def extend(
        self,
        replay_buffer: ReplayBuffer,
        replay_sampler: DreamerV3ReplaySampler,
        records: TensorDictBase,
    ) -> torch.Tensor:
        if records.ndim != 2 or records.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected replay records with shape [num_streams, time], got "
                f"{tuple(records.shape)} for {self.num_streams} streams."
            )
        self._finalize_tail(replay_buffer, records)
        placeholder = self._tail_placeholder(records)
        if self._tail_index is None:
            appended = torch.cat([records, placeholder], 1)
        else:
            appended = torch.cat([records[:, 1:], placeholder], 1)
        replay_indices = replay_buffer.extend(
            appended if self.num_streams > 1 else appended.reshape(-1)
        )
        replay_sampler.observe_extend(replay_indices, replay_buffer.storage)

        coordinates = torch.as_tensor(replay_indices, dtype=torch.long).reshape(
            appended.shape[1], self.num_streams, replay_buffer.storage.ndim
        )
        tail_index = coordinates[-1]
        if replay_buffer.storage.ndim == 1:
            tail_index = tail_index[:, 0]
        self._tail_index = tail_index.clone()
        self._tail_generation = replay_buffer.writer.generations_of(
            self._tail_index
        ).clone()
        return replay_indices


class MultiStreamReplay:
    """One single-stream replay buffer per environment stream.

    Asynchronous collection delivers the transitions of each environment in
    arrival order, with episode boundaries that differ across environments.
    Each stream therefore gets its own record builder, ring storage and
    sampler; a batch draws its sequences from the streams in proportion to
    the windows they hold.

    Args:
        num_streams (int): Number of environment streams.
        buffer_size (int): Records kept per stream.
        slice_len (int): Records per sampled sequence (the sequence length
            plus the extra context record).
        num_sequences (int): Sequences per sampled batch.
        online (bool): Serve the newest blocks of each stream before uniform
            samples, see :class:`DreamerV3ReplaySampler`.
        seed (int): Seed of the sampling streams.
        device (torch.device): Storage device.
        observation_keys (tuple of NestedKey, optional): See
            :class:`DreamerV3ReplayRecordBuilder`.
    """

    def __init__(
        self,
        num_streams: int,
        *,
        buffer_size: int,
        slice_len: int,
        num_sequences: int,
        online: bool,
        seed: int,
        device: torch.device,
        observation_keys: tuple[NestedKey, ...] = _DEFAULT_OBSERVATION_KEYS,
    ):
        if num_streams < 1:
            raise ValueError(f"num_streams must be positive, got {num_streams}.")
        self.num_streams = num_streams
        self.slice_len = slice_len
        self.num_sequences = num_sequences
        self.buffers = [
            ReplayBuffer(
                storage=LazyTensorStorage(max_size=buffer_size, ndim=1, device=device),
                writer=RoundRobinWriter(track_generations=True),
                sampler=DreamerV3ReplaySampler(slice_len=slice_len, online=online),
                batch_size=slice_len,
                generator=torch.Generator().manual_seed(seed + stream),
            )
            for stream in range(num_streams)
        ]
        self.builders = [
            DreamerV3ReplayRecordBuilder(1, observation_keys)
            for _ in range(num_streams)
        ]
        self._rng = torch.Generator().manual_seed(seed + num_streams)

    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self.buffers)

    def stream_lengths(self) -> list[int]:
        return [len(buffer) for buffer in self.buffers]

    @property
    def num_sampleable_streams(self) -> int:
        return sum(len(buffer) >= self.slice_len for buffer in self.buffers)

    def extend_stream(self, stream: int, data: TensorDictBase) -> int:
        """Append the collector transitions of one stream, returning the record count."""
        records = self.builders[stream](data).reshape(-1)
        buffer = self.buffers[stream]
        indices = buffer.extend(records)
        buffer.sampler.observe_extend(indices, buffer.storage)
        return records.numel()

    def sample(self, return_info: bool = True) -> tuple[TensorDictBase, dict]:
        """Sample ``num_sequences`` sequences across the streams."""
        weights = torch.tensor(
            [max(len(buffer) - self.slice_len + 1, 0) for buffer in self.buffers],
            dtype=torch.float,
        )
        if not weights.sum():
            raise RuntimeError(f"No replay stream holds {self.slice_len} records yet.")
        streams = torch.multinomial(
            weights, self.num_sequences, replacement=True, generator=self._rng
        )
        counts = torch.bincount(streams, minlength=self.num_streams)
        parts = []
        infos = []
        for stream in counts.nonzero().flatten().tolist():
            count = int(counts[stream])
            data, info = self.buffers[stream].sample(
                batch_size=count * self.slice_len, return_info=True
            )
            parts.append(data)
            infos.append((stream, count, info))
        data = torch.cat(parts, 0)
        if not return_info:
            return data
        return data, {"streams": infos}

    def refresh_context(
        self, info: dict, state: torch.Tensor, belief: torch.Tensor
    ) -> None:
        """Write refreshed latents back, one stream at a time."""
        offset = 0
        for stream, count, part_info in info["streams"]:
            _refresh_replay_context(
                self.buffers[stream],
                part_info["index"],
                part_info["index_generation"],
                state[offset : offset + count],
                belief[offset : offset + count],
            )
            offset += count


class DreamerV3ReplaySampler(SliceSampler):
    """Slice sampler that takes the oldest queued blocks before uniform ones."""

    def __init__(self, *args, online: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.online = online
        self._stream_lengths: torch.Tensor | None = None
        self._online_queue: list[torch.Tensor] = []

    @property
    def online_queue_size(self) -> int:
        return len(self._online_queue)

    def observe_extend(self, index: torch.Tensor, storage: LazyTensorStorage) -> None:
        """Queue the start of each new, non-overlapping ``slice_len`` block."""
        if not self.online:
            return
        index = torch.as_tensor(index, dtype=torch.long)
        if storage.ndim == 1:
            coordinates = index.reshape(-1, 1, 1)
            num_streams = 1
        else:
            num_streams = storage.shape[1:].numel()
            coordinates = index.reshape(-1, num_streams, storage.ndim)
        if self._stream_lengths is None:
            self._stream_lengths = torch.zeros(num_streams, dtype=torch.long)
        elif self._stream_lengths.numel() != num_streams:
            raise RuntimeError(
                "The number of replay streams changed after initialization."
            )

        max_time = storage._max_size_along_dim0()
        num_rows = coordinates.shape[0]
        if not num_rows:
            return
        # Every stream grows by one record per row, so the stream lengths stay
        # equal: a row completes a block when the shared length passes a
        # multiple of slice_len.
        lengths = self._stream_lengths[0] + torch.arange(1, num_rows + 1)
        completes_block = (lengths > self.slice_len) & (
            (lengths - 1).remainder(self.slice_len) == 0
        )
        self._stream_lengths.add_(num_rows)
        if completes_block.any():
            starts = coordinates[completes_block].clone()
            starts[..., 0].sub_(self.slice_len - 1).remainder_(max_time)
            self._online_queue.extend(
                starts.reshape(-1, coordinates.shape[-1]).unbind(0)
            )

    def _drop_stale_online(self, storage: LazyTensorStorage, seq_length: int) -> None:
        """Drop queued starts whose ``seq_length`` window is no longer stored."""
        if not self._online_queue or not storage._is_full:
            return
        stored_time = storage.shape[0]
        oldest = (int(storage._last_cursor_index) + 1) % stored_time
        live = stored_time - seq_length + 1
        self._online_queue = [
            start
            for start in self._online_queue
            if (int(start[0]) - oldest) % stored_time < live
        ]

    def sample(
        self, storage: LazyTensorStorage, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], dict]:
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        self._drop_stale_online(storage, seq_length)
        # Each sequence of a batch takes one online block if the queue has one.
        num_online = min(num_slices, len(self._online_queue))
        num_uniform = num_slices - num_online
        if storage.ndim > 2:
            raise RuntimeError("DreamerV3 continuous replay supports 1D or 2D storage.")
        if num_uniform:
            stored_time = storage.shape[0]
            num_starts = stored_time - seq_length + 1
            if num_starts < 1:
                raise RuntimeError(
                    f"Replay streams have length {stored_time}, but sampling "
                    f"requires {seq_length} records."
                )
            num_streams = 1 if storage.ndim == 1 else storage.shape[1]
            flat_start = torch.randint(
                num_starts * num_streams,
                (num_uniform,),
                generator=self._rng,
            )
            relative_time = flat_start.div(num_streams, rounding_mode="floor")
            stream = flat_start.remainder(num_streams)
            oldest_time = (
                (int(storage._last_cursor_index) + 1) % stored_time
                if storage._is_full
                else 0
            )
            start_time = (relative_time + oldest_time).remainder(stored_time)
            if storage.ndim == 1:
                uniform_starts = start_time.unsqueeze(-1)
            else:
                uniform_starts = torch.stack([start_time, stream], -1)
            uniform_coordinates = self._tensor_slices_from_startend(
                seq_length,
                uniform_starts,
                stored_time,
            ).reshape(num_uniform, seq_length, storage.ndim)
            index_device = uniform_starts.device
        else:
            uniform_coordinates = None
            index_device = self._online_queue[0].device

        if num_online:
            online_starts = torch.stack(
                [self._online_queue.pop(0) for _ in range(num_online)]
            ).to(index_device)
            online_coordinates = self._tensor_slices_from_startend(
                seq_length,
                online_starts,
                storage.shape[0],
            ).reshape(num_online, seq_length, storage.ndim)
        else:
            online_coordinates = None
        coordinates = torch.cat(
            [
                candidate
                for candidate in (online_coordinates, uniform_coordinates)
                if candidate is not None
            ],
            0,
        )
        return coordinates.reshape(-1, storage.ndim).unbind(-1), {}

    def _empty(self) -> None:
        super()._empty()
        self._stream_lengths = None
        self._online_queue.clear()
