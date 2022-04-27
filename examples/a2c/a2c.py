from torchrl.agents.helpers.collectors import \
    make_collector_offpolicy_singleprocess
from torchrl.modules.td_module.exploration import StateLessEGreedyWrapper

try:
    import configargparse as argparse

    _configargparse = True
except ImportError:
    import argparse

    _configargparse = False
import os
import sys
import tempfile
import uuid
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torchrl.agents.helpers import transformed_env_constructor, make_dqn_actor, \
    make_dqn_loss, get_stats_random_rollout, parallel_env_constructor, \
    make_collector_offpolicy, make_agent, parser_agent_args, \
    parser_collector_args_offpolicy, parser_loss_args, parser_env_args, \
    parser_model_args_discrete, parser_recorder_args, parser_replay_args
from torchrl.envs import TransformedEnv, RewardScaling
from torchrl.modules import EGreedyWrapper


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def A2C(rank, world_size, args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # some setup
    if not isinstance(args.reward_scaling, float):
        args.reward_scaling = 1.0

    device = torch.device(f"cuda:{rank}")
    args.collector_devices = [device]
    args.total_frames = args.total_frames // world_size

    replay_buffer = None  # no RB for A2C

    if rank == 0:
        exp_name = "_".join(
            [
                "A2C",
                args.exp_name,
                str(uuid.uuid4())[:8],
                datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
            ]
        )
        writer = SummaryWriter(f"dqn_logging/{exp_name}")
        video_tag = exp_name if args.record_video else ""
    else:
        writer = video_tag = exp_name = None

    # make the env creator function
    proof_env = transformed_env_constructor(args=args, use_env_creator=False)()
    model = make_dqn_actor(
        proof_environment=proof_env,
        args=args,
        device=device,
    )

    # make the policy
    if args.eps_greedy:
        model_explore = StateLessEGreedyWrapper(
            model,
        ).to(device)
    else:
        model_explore = model

    # make the data collector (single process)
    stats = None
    if not args.vecnorm:
        stats = get_stats_random_rollout(args, proof_env)
    # make sure proof_env is closed
    proof_env.close()

    if args.vecnorm:
        raise NotImplementedError(
            "VecNorm is currently not supported in A2C. Work plan:"
            "\n- Implement a communication pipeline between processes to share norm constants, "
            "\n- Write tests for this with RPC"
        )

    create_env_fn = parallel_env_constructor(args=args, stats=stats)

    collector = make_collector_offpolicy_singleprocess(
        make_env=create_env_fn,
        actor_model_explore=model_explore,
        args=args,
    )

    # make the logger
    if rank == 0:
        recorder = transformed_env_constructor(
            args,
            video_tag=video_tag,
            norm_obs_only=True,
            stats=stats,
            writer=writer,
        )()
        # remove video recorder from recorder to have matching state_dict keys
        if args.record_video:
            recorder_rm = TransformedEnv(recorder.env, recorder.transform[1:])
        else:
            recorder_rm = recorder
        recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
        # reset reward scaling
        for t in recorder.transform:
            if isinstance(t, RewardScaling):
                t.scale.fill_(1.0)

    else:
        recorder = None


    # make the loss
    loss_module, target_net_updater = make_dqn_loss(model, args)
    loss_module = DDP(loss_module, device_ids=[rank])

    # make the agent
    agent = make_agent(
        collector,
        loss_module,
        recorder,
        target_net_updater,
        model_explore,
        replay_buffer,
        writer,
        args,
        world_size=world_size,
        progress_bar=rank==0,
    )

    # run the agent
    agent.train()

    # cleanup
    cleanup()


def run_a2c(a2c_fn, world_size, args):
    mp.spawn(a2c_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

def make_args():
    parser = argparse.ArgumentParser()
    if _configargparse:
        parser.add_argument(
            "-c",
            "--config",
            required=True,
            is_config_file=True,
            help="config file path",
        )
    parser_agent_args(parser)
    parser_collector_args_offpolicy(parser)
    parser_env_args(parser)
    parser_loss_args(parser, algorithm="A2C")
    parser_model_args_discrete(parser)
    parser_recorder_args(parser)
    parser_replay_args(parser)
    return parser

parser = make_args()

if __name__ == "__main__":
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_a2c(A2C, world_size, args)
