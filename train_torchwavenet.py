import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import asdict


import socketserver
import sys

from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.cuda import device_count
from torch.multiprocessing import spawn
from typing import List

from src.config_torchwavenet import Config, parse_configs
from src.learner_torchwavenet import train, train_distributed
import wandb
#from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


def _get_free_port():
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

all_datasets = ['OFDMQPSK']

def main(argv: List[str]):
    parser = ArgumentParser(description="Train a Diffwave model.")
    parser.add_argument("--sigindex", type=int, default=0,
                        help="Index for Mixture Type.")
    parser.add_argument("--config", type=str, default="src/configs/wavenet.yml",
                        help="Configuration file for model.")
    parser.add_argument("--M", type=int, default=50,
                        help="Num of symbols.")                    
    args = parser.parse_args(argv[1:-1])
    
    sigtype = all_datasets[args.sigindex]
   
    # First create the base config
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(
        argv[-1].split("::")) if argv[-1] != "" else None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))
    cfg.data.root_dir = f"npydataset/Dataset_{sigtype}_train_Mixture_M={args.M}"
    cfg.model_dir = f"torchmodels/dataset_{sigtype.lower()}_mixture_wavenet"
    
    # Setup training
    wandb.init(
        project="rfchallenge-wavenet",
        name   = f"{sigtype}_M{args.M}",
        config=asdict(cfg),
        settings=wandb.Settings(start_method="thread")
    )
    wandb.watch_called = False 
    world_size = device_count()
    if world_size != cfg.distributed.world_size:
        raise ValueError(
            "Requested world size is not the same as number of visible GPUs.")
    if cfg.distributed.distributed:
        if world_size < 2:
            raise ValueError(
                "Distributed training cannot be run on machine"
                f" with {world_size} device(s)."
            )
        if cfg.data.batch_size % world_size != 0:
            raise ValueError(
                f"Batch size {cfg.data.batch_size} is not evenly"
                f" divisble by # GPUs = {world_size}."
            )
        cfg.data.batch_size = cfg.data.batch_size // world_size
        port = _get_free_port()
        spawn(
            train_distributed, 
            args=(world_size, port, cfg), 
            nprocs=world_size, 
            join=True
        )
    else:
        train(cfg)
    wandb.finish()    


if __name__ == "__main__":
    argv = sys.argv
    if len(sys.argv) == 3:
        argv = argv + [""]
    main(argv)
