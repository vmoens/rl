# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration
and a proprioceptive DeepMind Control Walker Walk reproduction configuration.

Run the small example with:

```bash
python sota-implementations/dreamer_v3/train.py
```

Run the full Walker Walk configuration with:

```bash
python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_walker
```

The Walker preset tracks the author-maintained JAX implementation at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. It matches that implementation's
640,867-parameter `size1m` configuration, uses 16 environments, batches of 16
sequences of length 64, a replay ratio of 1024, and 1.1 million environment
steps. BF16 training is enabled on CUDA. It logs stochastic training-episode
returns against environment steps, matching the current JAX curve protocol
without relying on wall-clock-dependent training iterations.

The Walker task is seeded from `env.seed`, as every other TorchRL example is;
pass `env.use_seed=false` for the JAX implementation's unseeded DMC resets. The
step axis counts initial and reset-only driver records as that
implementation does.

This is deliberately a reproduction of the pinned JAX `dmc_proprio` preset,
not of the paper's proprioceptive protocol. The two protocols differ:

| Setting | Pinned JAX `dmc_proprio` preset | DreamerV3 paper proprioceptive protocol |
| --- | --- | --- |
| Model size | `size1m` (640,867 parameters here) | 12M parameters |
| Environment steps | 1.1M | 500K |
| Action repeat | 1 | 2 |
| Replay ratio | 1024 | 512 |
| Optimizer | AGC, LaProp-style RMS scaling then momentum, 1,000-step warmup | Paper recipe |
| Reported aggregation | Three-seed median and interquartile range in this benchmark | Five-seed mean and standard deviation |

TorchRL's public DreamerV3 API documentation remains centered on the paper's
algorithmic semantics. This named SOTA preset documents later choices in the
evolving JAX codebase instead of silently treating them as paper requirements.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

For a three-seed median and interquartile reproduction run:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh
```

For the fastest supported accelerator path, enable the compiled RSSM scan
(unrolled eight steps at a time) and CUDA-graph capture of the fixed-shape
learner forward/backward:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh --fast
```

To measure the same fixed-shape learner update after compile and capture
warmup—including every loss, backward, optimizer, and slow-target update—run:

```bash
python benchmarks/ad_hoc/bench_dreamer_v3_learner.py
```

The timing excludes replay sampling and environment collection. Use the
benchmark arguments to change the batch size, sequence length, scan unroll,
warmup, or number of measured updates. Compilation and graph capture happen
during warmup and are excluded from the reported samples.

On one NVIDIA GB200 with PyTorch 2.12.0, CUDA 13.0, BF16, batch size 16,
sequence length 64, scan unroll 8, 10 warmup updates and 50 measured updates:

| Learner backend | Median update | Transitions/s | Speedup |
| --- | ---: | ---: | ---: |
| Compiled scan | 358.51 ms | 2,856 | 1.00x |
| Compiled scan + CUDA graph | 17.83 ms | 57,415 | 20.10x |

Compilation has an up-front cost, so the short validation remains eager:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh --smoke
```

The benchmark writes one metrics file per seed plus `summary.json`, aggregates
the stochastic training returns into median and interquartile curves over fixed
windows, and fails when the final window median falls short. The seeds, the
window and the threshold come from the `benchmark` block of
`config_dmc_walker.yaml`, which ships three seeds, 50,000-step windows and a
minimum final median return of 900; `benchmark.*` Hydra overrides change them,
as in `benchmark.seeds=[0,1,2,3,4]`. `env.seed` and `logger.metrics_jsonl` are
set per run and are rejected as overrides, since either would collapse the
seeds onto one trajectory. Full learning-curve runs are intended for scheduled
or manual validation; pull-request CI uses short smoke overrides. Set
`OUTPUT_DIR` to change the output directory (the defaults are
`dmc_walker_runs` and `dmc_walker_smoke`), and append any other Hydra overrides
to the wrapper, for example `benchmark.seeds=[0]`. Each run logs the resolved
training device, replay device, RSSM backend, scan unroll, mixed-precision state
and learner CUDA-graph setting.

For a smaller ablation, shorten the run rather than the window:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir smoke \
  collector.total_frames=100000 \
  benchmark.minimum_final_median_return=0
```

Every worker runs to the same time limit, so episodes finish in bursts one
episode apart: `(env.max_episode_steps + 1) * collector.num_envs`, or 16,016
records for the preset. A window narrower than that holds no completed episode
over most of the run, so the script refuses one before launching anything. The
command above keeps the 50,000-step window and still fills two of them with
about 48 episodes each.

`optimization.compile_rssm` compiles the RSSM recurrence and is off by default,
since a short run never repays the build. `step` compiles the deterministic work
and draws the same categories as an eager run; `scan` compiles the unrolled
recurrence and the imagination prior, and is faster, but its draws fall inside
the compiled region, so a seeded run diverges from an eager one. The scan uses
`optimization.rssm_scan_unroll=8` by default; lower values reduce compilation
time and graph size, while `1` disables manual unrolling.
`optimization.cudagraph_train_step=true` captures the learner forward and
backward after five warmup calls. It requires CUDA and fixed input shapes;
optimizer and target-network steps remain outside capture so their schedules
continue to advance normally. With either option on, the learner step is
compiled and captured on a spec-shaped fake batch before collection starts
(`optimization.compile_warmup`, on by default in that case); parameters,
normalizer statistics and target networks are restored afterwards, and the
optimizer does not step. Compiling while the asynchronous collector runs would
otherwise share the interpreter with its threads and idle the environments for
the whole compile.

## Images, discrete actions and asynchronous collection

The script also trains from images, from a flat vector, or from both. `env.vector_key`
names the flat observation (symlog MLP encoder, symlog squared-error decoder) and
`env.pixels_key` a channels-first `uint8` image (`DreamerV3ImageEncoder` /
`DreamerV3ImageDecoder`, squared error on the image divided by 255, sized by
`networks.image_*`). A `OneHot` action spec selects a categorical actor with the
reference's 1% uniform mixture; continuous actions keep the tanh-normal actor.

`env.backend=custom` plugs in any environment through an import path,
`env.factory=package.module:function`, called as
`factory(seed=..., env_index=..., num_envs=..., **env.factory_kwargs)` and
followed by the usual `StepCounter` and `InitTracker`.

`collector.backend=async` replaces the synchronous `Collector` with an
`AsyncBatchedCollector`: `collector.num_envs` environments step independently in
their own processes (`collector.env_backend`) while an inference server on
`collector.policy_device` batches the policy calls
(`collector.inference_max_batch_size`, `collector.inference_timeout`). Since the
environments no longer share a time axis, replay keeps one ring per environment
(`replay_buffer.buffer_size` records each) and batches draw their sequences across
the rings. `collector.max_pending_frames` bounds the frames waiting for the
learner so environments pause when it lags, like a samples-per-insert limiter;
`optimization.deferred_policy_sync=true` is required because the server owns a
copy of the policy that is refreshed between batches. Environments that expose a
boolean `env.milestone_key` vector have its flags logged at the end of every
episode under `env.milestone_names`.

`optimization.collection_warmup_seconds` delays the first update so the
environments and the inference server run unthrottled for that long, which
measures their throughput before the train ratio couples collection to the
learner; `collector.compile_policy=true` compiles the acting policy and traces
it for the served batch sizes before collection starts, falling back to the
eager policy if compilation fails.

`optimization.max_time` stops a run after a wall-clock budget (with
`collector.total_frames=-1` for an unbounded frame budget), and
`logger.backend=wandb` mirrors every JSONL record to Weights & Biases
(`logger.project`, `logger.entity`, `logger.exp_name`, and `logger.base_url` to
pin the server when an imported library redirects `WANDB_BASE_URL`), including the collection
rates, the inference-server statistics and the learner timings that the `train`
records carry.

## Checkpoint and resume

The trainer writes a final checkpoint on a normal, time-budget, interrupt, or
termination-signal shutdown and writes periodic checkpoints every
`optimization.checkpoint_every` learner updates. Checkpoints rotate below
`optimization.checkpoint_dir`; set that directory to `null` to disable saves.
On restart, `optimization.resume_from` accepts either one checkpoint or a
checkpoint directory. When it is `null`, the newest checkpoint already in the
configured checkpoint directory is used automatically.

Learner and target parameters, optimizer state, return normalization, RNG
streams, replay cursors, cumulative environment/action/update counters and
elapsed time are restored. The cumulative environment-step counter remains the
step axis for JSONL and W&B metrics. A resumed W&B logger uses the saved run ID
with `resume="must"`.

Replay tensors are excluded by default because they can dominate checkpoint
size. Set `optimization.checkpoint_include_replay=true` to save and restore the
synchronous replay buffer or every asynchronous per-environment replay buffer,
including their sampler, writer and latent-context state. Without that option,
the resumed learner keeps its progress while replay safely refills from empty.
