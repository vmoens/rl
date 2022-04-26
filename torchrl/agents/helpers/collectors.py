# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace
from typing import Callable, List, Optional, Type, Union

from torchrl.collectors.collectors import (
    _DataCollector,
    _MultiDataCollector,
    MultiaSyncDataCollector,
    MultiSyncDataCollector, SyncDataCollector,
)
from torchrl.data import MultiStep
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs import ParallelEnv

__all__ = [
    "sync_sync_collector",
    "sync_async_collector",
    "make_collector_offpolicy",
    "make_collector_onpolicy",
    "parser_collector_args_offpolicy",
    "parser_collector_args_onpolicy",
]

from torchrl.envs.common import _EnvClass
from torchrl.modules import ProbabilisticTDModule, TDModuleWrapper


def sync_async_collector(
    env_fns: Union[Callable, List[Callable]],
    env_kwargs: Optional[Union[dict, List[dict]]],
    num_env_per_collector: Optional[int] = None,
    num_collectors: Optional[int] = None,
    **kwargs,
) -> MultiaSyncDataCollector:
    """
    Runs asynchronous collectors, each running synchronous environments.

    .. aafig::


            +----------------------------------------------------------------------+
            |           "MultiConcurrentCollector"                |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |  "Collector 1"  |  "Collector 2"  |  "Collector 3"  |     "Main"     |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor    | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "yield batch 1" |       "actor"   |                 |"collect, train"|
            |                 |                 |                 |                |
            | "step" | "step" |                 | "yield batch 2" |"collect, train"|
            |        |        |                 |                 |                |
            |        |        | "yield batch 3" |                 |"collect, train"|
            |        |        |                 |                 |                |
            +----------------------------------------------------------------------+

    Environment types can be identical or different. In the latter case, env_fns should be a list with all the creator
    fns for the various envs,
    and the policy should handle those envs in batch.

    Args:
        env_fns: Callable (or list of Callables) returning an instance of _EnvClass class.
        env_kwargs: Optional. Dictionary (or list of dictionaries) containing the kwargs for the environment being created.
        num_env_per_collector: Number of environments per data collector. The product
            num_env_per_collector * num_collectors should be less or equal to the number of workers available.
        num_collectors: Number of data collectors to be run in parallel.
        **kwargs: Other kwargs passed to the data collectors

    """

    return _make_collector(
        MultiaSyncDataCollector,
        env_fns=env_fns,
        env_kwargs=env_kwargs,
        num_env_per_collector=num_env_per_collector,
        num_collectors=num_collectors,
        **kwargs,
    )


def sync_sync_collector(
    env_fns: Union[Callable, List[Callable]],
    env_kwargs: Optional[Union[dict, List[dict]]],
    num_env_per_collector: Optional[int] = None,
    num_collectors: Optional[int] = None,
    **kwargs,
) -> MultiSyncDataCollector:
    """
    Runs synchronous collectors, each running synchronous environments.

    E.g.

    .. aafig::

            +----------------------------------------------------------------------+
            |            "MultiConcurrentCollector"               |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |   "Collector 1" |  "Collector 2"  |  "Collector 3"  |     Main       |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            |                 |       "actor"   |                 |                |
            |                 |                 |                 |                |
            |                       "yield batch of traj 1"------->"collect, train"|
            |                                                     |                |
            | "step" | "step" | "step" | "step" | "step" | "step" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |       "actor"   |        |        |                |
            |                 | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   |                 |       "actor"   |                |
            |                       "yield batch of traj 2"------->"collect, train"|
            |                                                     |                |
            +----------------------------------------------------------------------+

    Envs can be identical or different. In the latter case, env_fns should be a list with all the creator fns
    for the various envs,
    and the policy should handle those envs in batch.

    Args:
        env_fns: Callable (or list of Callables) returning an instance of _EnvClass class.
        env_kwargs: Optional. Dictionary (or list of dictionaries) containing the kwargs for the environment being created.
        num_env_per_collector: Number of environments per data collector. The product
            num_env_per_collector * num_collectors should be less or equal to the number of workers available.
        num_collectors: Number of data collectors to be run in parallel.
        **kwargs: Other kwargs passed to the data collectors

    """
    return _make_collector(
        MultiSyncDataCollector,
        env_fns=env_fns,
        env_kwargs=env_kwargs,
        num_env_per_collector=num_env_per_collector,
        num_collectors=num_collectors,
        **kwargs,
    )


def _make_collector(
    collector_class: Type,
    env_fns: Union[Callable, List[Callable]],
    env_kwargs: Optional[Union[dict, List[dict]]],
    policy: Callable[[_TensorDict], _TensorDict],
    max_frames_per_traj: int = -1,
    frames_per_batch: int = 200,
    total_frames: Optional[int] = None,
    postproc: Optional[Callable] = None,
    num_env_per_collector: Optional[int] = None,
    num_collectors: Optional[int] = None,
    **kwargs,
) -> _MultiDataCollector:
    if env_kwargs is None:
        env_kwargs = dict()
    if isinstance(env_fns, list):
        num_env = len(env_fns)
        if num_env_per_collector is None:
            num_env_per_collector = -(num_env // -num_collectors)
        elif num_collectors is None:
            num_collectors = -(num_env // -num_env_per_collector)
        else:
            if num_env_per_collector * num_collectors < num_env:
                raise ValueError(
                    f"num_env_per_collector * num_collectors={num_env_per_collector * num_collectors} "
                    f"has been found to be less than num_env={num_env}"
                )
    else:
        try:
            num_env = num_env_per_collector * num_collectors
            env_fns = [env_fns for _ in range(num_env)]
        except (TypeError):
            raise Exception(
                "num_env was not a list but num_env_per_collector and num_collectors were not both specified,"
                f"got num_env_per_collector={num_env_per_collector} and num_collectors={num_collectors}"
            )
    if not isinstance(env_kwargs, list):
        env_kwargs = [env_kwargs for _ in range(num_env)]

    env_fns_split = [
        env_fns[i : i + num_env_per_collector]
        for i in range(0, num_env, num_env_per_collector)
    ]
    env_kwargs_split = [
        env_kwargs[i : i + num_env_per_collector]
        for i in range(0, num_env, num_env_per_collector)
    ]
    if len(env_fns_split) != num_collectors:
        raise RuntimeError(
            f"num_collectors={num_collectors} differs from len(env_fns_split)={len(env_fns_split)}"
        )

    if num_env_per_collector == 1:
        env_fns = [_env_fn[0] for _env_fn in env_fns_split]
        env_kwargs = [_env_kwargs[0] for _env_kwargs in env_kwargs_split]
    else:
        env_fns = [
            lambda: ParallelEnv(
                num_workers=len(_env_fn),
                create_env_fn=_env_fn,
                create_env_kwargs=_env_kwargs,
            )
            for _env_fn, _env_kwargs in zip(env_fns_split, env_kwargs_split)
        ]
        env_kwargs = None
    return collector_class(
        create_env_fn=env_fns,
        create_env_kwargs=env_kwargs,
        policy=policy,
        total_frames=total_frames,
        max_frames_per_traj=max_frames_per_traj,
        frames_per_batch=frames_per_batch,
        postproc=postproc,
        **kwargs,
    )


def make_collector_offpolicy(
    make_env: Callable[[], _EnvClass],
    actor_model_explore: Union[TDModuleWrapper, ProbabilisticTDModule],
    args: Namespace,
    make_env_kwargs=None,
) -> _DataCollector:
    """
    Returns a data collector for off-policy algorithms.

    Args:
        make_env (Callable): environment creator
        actor_model_explore (TDModule): Model instance used for evaluation and exploration update
        args (Namespace): argument namespace built from the parser constructor
        make_env_kwargs (dict): kwargs for the env creator

    """
    if args.async_collection:
        collector_helper = sync_async_collector
    else:
        collector_helper = sync_sync_collector

    if args.multi_step:
        ms = MultiStep(
            gamma=args.gamma,
            n_steps_max=args.n_steps_return,
        )
    else:
        ms = None

    env_kwargs = {}
    if make_env_kwargs is not None:
        env_kwargs.update(make_env_kwargs)
    args.collector_devices = (
        args.collector_devices
        if len(args.collector_devices) > 1
        else args.collector_devices[0]
    )
    collector_helper_kwargs = {
        "env_fns": make_env,
        "env_kwargs": env_kwargs,
        "policy": actor_model_explore,
        "max_frames_per_traj": args.max_frames_per_traj,
        "frames_per_batch": args.frames_per_batch,
        "total_frames": args.total_frames,
        "postproc": ms,
        "num_env_per_collector": 1,
        # we already took care of building the make_parallel_env function
        "num_collectors": -args.num_workers // -args.env_per_collector,
        "devices": args.collector_devices,
        "passing_devices": args.collector_devices,
        "init_random_frames": args.init_random_frames,
        "pin_memory": args.pin_memory,
        "split_trajs": ms is not None,
        # trajectories must be separated if multi-step is used
        "init_with_lag": args.init_with_lag,
        "exploration_mode": args.exploration_mode,
    }

    collector = collector_helper(**collector_helper_kwargs)
    collector.set_seed(args.seed)
    return collector

def make_collector_offpolicy_singleprocess(
    make_env: Callable[[], _EnvClass],
    actor_model_explore: Union[TDModuleWrapper, ProbabilisticTDModule],
    args: Namespace,
    make_env_kwargs=None,
) -> _DataCollector:
    """
    Returns a data collector for off-policy algorithms.

    Args:
        make_env (Callable): environment creator
        actor_model_explore (TDModule): Model instance used for evaluation and exploration update
        args (Namespace): argument namespace built from the parser constructor
        make_env_kwargs (dict): kwargs for the env creator

    """
    if args.multi_step:
        ms = MultiStep(
            gamma=args.gamma,
            n_steps_max=args.n_steps_return,
        )
    else:
        ms = None

    env_kwargs = {}
    if make_env_kwargs is not None:
        env_kwargs.update(make_env_kwargs)
    args.collector_devices = args.collector_devices[0]
    collector_helper_kwargs = {
        "create_env_fn": make_env,
        "env_kwargs": env_kwargs,
        "policy": actor_model_explore,
        "max_frames_per_traj": args.max_frames_per_traj,
        "frames_per_batch": args.frames_per_batch,
        "total_frames": args.total_frames,
        "postproc": ms,
        "device": args.collector_devices,
        "passing_devices": args.collector_devices,
        "init_random_frames": args.init_random_frames,
        "pin_memory": args.pin_memory,
        "split_trajs": ms is not None,
        # trajectories must be separated if multi-step is used
        "init_with_lag": args.init_with_lag,
        "exploration_mode": args.exploration_mode,
    }

    collector = SyncDataCollector(**collector_helper_kwargs)
    collector.set_seed(args.seed)
    return collector


def make_collector_onpolicy(
    make_env: Callable[[], _EnvClass],
    actor_model_explore: Union[TDModuleWrapper, ProbabilisticTDModule],
    args: Namespace,
    make_env_kwargs=None,
) -> _DataCollector:
    collector_helper = sync_sync_collector

    ms = None

    env_kwargs = {}
    if make_env_kwargs is not None:
        env_kwargs.update(make_env_kwargs)
    args.collector_devices = (
        args.collector_devices
        if len(args.collector_devices) > 1
        else args.collector_devices[0]
    )
    collector_helper_kwargs = {
        "env_fns": make_env,
        "env_kwargs": env_kwargs,
        "policy": actor_model_explore,
        "max_frames_per_traj": args.max_frames_per_traj,
        "frames_per_batch": args.frames_per_batch,
        "total_frames": args.total_frames,
        "postproc": ms,
        "num_env_per_collector": 1,
        # we already took care of building the make_parallel_env function
        "num_collectors": -args.num_workers // -args.env_per_collector,
        "devices": args.collector_devices,
        "passing_devices": args.collector_devices,
        "pin_memory": args.pin_memory,
        "split_trajs": True,
        # trajectories must be separated in online settings
        "init_with_lag": args.init_with_lag,
        "exploration_mode": args.exploration_mode,
    }

    collector = collector_helper(**collector_helper_kwargs)
    collector.set_seed(args.seed)
    return collector


def _parser_collector_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--collector_devices",
        "--collector-devices",
        nargs="+",
        default=["cpu"],
        help="device on which the data collector should store the trajectories to be passed to this script."
        "If the collector device differs from the policy device (cuda:0 if available), then the "
        "weights of the collector policy are synchronized with collector.update_policy_weights_().",
    )
    parser.add_argument(
        "--pin_memory",
        "--pin-memory",
        action="store_true",
        help="if True, the data collector will call pin_memory before dispatching tensordicts onto the "
        "passing device.",
    )
    parser.add_argument(
        "--init_with_lag",
        "--init-with-lag",
        action="store_true",
        help="if True, the first trajectory will be truncated earlier at a random step. This is helpful"
        " to desynchronize the environments, such that steps do no match in all collected "
        "rollouts. Especially useful for online training, to prevent cyclic sample indices.",
    )
    parser.add_argument(
        "--frames_per_batch",
        "--frames-per-batch",
        type=int,
        default=1000,
        help="number of steps executed in the environment per collection."
        "This value represents how many steps will the data collector execute and return in *each*"
        "environment that has been created in between two rounds of optimization "
        "(see the optim_steps_per_collection above). "
        "On the one hand, a low value will enhance the data throughput between processes in async "
        "settings, which can make the accessing of data a computational bottleneck. "
        "High values will on the other hand lead to greater tensor sizes in memory and disk to be "
        "written and read at each global iteration. One should look at the number of frames per second"
        "in the log to assess the efficiency of the configuration.",
    )
    parser.add_argument(
        "--total_frames",
        "--total-frames",
        type=int,
        default=50000000,
        help="total number of frames collected for training. Does account for frame_skip (i.e. will be "
        "divided by the frame_skip). Default=50e6.",
    )
    parser.add_argument(
        "--num_workers",
        "--num-workers",
        type=int,
        default=32,
        help="Number of workers used for data collection. ",
    )
    parser.add_argument(
        "--env_per_collector",
        "--env-per-collector",
        default=8,
        type=int,
        help="Number of environments per collector. If the env_per_collector is in the range: "
        "1<env_per_collector<=num_workers, then the collector runs"
        "ceil(num_workers/env_per_collector) in parallel and executes the policy steps synchronously "
        "for each of these parallel wrappers. If env_per_collector=num_workers, no parallel wrapper is created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for the environment, pytorch and numpy.",
    )
    parser.add_argument(
        "--exploration_mode",
        type=str,
        default=None,
        help="exploration mode of the data collector. If gSDE is being used, this should be set to `'net_output'`.",
    )
    parser.add_argument(
        "--async_collection",
        "--async-collection",
        action="store_true",
        help="whether data collection should be done asynchrously. Asynchrounous data collection means "
        "that the data collector will keep on running the environment with the previous weights "
        "configuration while the optimization loop is being done. If the algorithm is trained "
        "synchronously, data collection and optimization will occur iteratively, not concurrently.",
    )
    return parser


def parser_collector_args_offpolicy(parser: ArgumentParser) -> ArgumentParser:
    """
    Populates the argument parser to build a data collector for on-policy algorithms (DQN, DDPG, SAC, REDQ).

    Args:
        parser (ArgumentParser): parser to be populated.

    """
    parser = _parser_collector_args(parser)
    parser.add_argument(
        "--multi_step",
        "--multi-step",
        dest="multi_step",
        action="store_true",
        help="whether or not multi-step rewards should be used.",
    )
    parser.add_argument(
        "--n_steps_return",
        "--n-steps-return",
        type=int,
        default=3,
        help="If multi_step is set to True, this value defines the number of steps to look ahead for the "
        "reward computation.",
    )
    parser.add_argument(
        "--init_random_frames",
        "--init-random-frames",
        type=int,
        default=50000,
        help="Initial number of random frames used before the policy is being used. Default=5000.",
    )
    return parser


def parser_collector_args_onpolicy(parser: ArgumentParser) -> ArgumentParser:
    """
    Populates the argument parser to build a data collector for on-policy algorithms (PPO).

    Args:
        parser (ArgumentParser): parser to be populated.
    """
    parser = _parser_collector_args(parser)
    return parser
