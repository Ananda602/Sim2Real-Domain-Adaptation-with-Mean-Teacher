# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`

import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(
        description="Semi-Supervised Segmentation with Two-Teacher Offline Distillation"
    )

    # ----------------------------
    # Data
    # ----------------------------
    data = parser.add_argument_group("data")
    data.add_argument("--real-root", type=str, required=False,
                      help="Root folder of REAL dataset (preferred).")
    data.add_argument("--sim-root", type=str, default=None,
                      help="Root folder of SIMULATED dataset (optional).")
    data.add_argument("--real-task", type=str, default=None,
                      help="REAL task folder name under each part (default: tree).")
    data.add_argument("--sim-tasks", type=str, default=None,
                      help="Comma-separated SIM tasks (default: whole_tree,tree_trunk).")
    data.add_argument("--train-split", type=str, default=None,
                      help="Train split name (default: train).")
    data.add_argument("--eval-split", type=str, default=None,
                      help="Eval split name (default: val).")
    data.add_argument("--img-size", type=int, default=None,
                      help="Square resize size fed to the model (default: 128).")

    # ----------------------------
    # Model
    # ----------------------------
    model = parser.add_argument_group("model")
    model.add_argument("--num-classes", type=int, default=None,
                       help="#classes for segmentation (default: 1).")
    model.add_argument("--pretrained", action="store_true",
                       help="Use pretrained weights when factory supports it.")
    model.add_argument("--arch", default=None, help="student architecture")
    model.add_argument("--arch-teacher", default=None, help="teacher architecture")
    model.add_argument("--teacher-cfg", default=None, help="teacher config")
    model.add_argument("--config",type=str,default=None, help="Path to YAML config file (overrides defaults, can be overridden by CLI args)")

    # ----------------------------
    # Optimization
    # ----------------------------
    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=None,
                       help="Total epochs (default: 100).")
    optim.add_argument("--start-epoch", type=int, default=None,
                       help="Start epoch (for resume).")
    optim.add_argument("-b", "--batch-size", type=int, default=None,
                       help="Total batch size (default: 8).")
    optim.add_argument("--labeled-batch-size", type=int, default=None,
                       help="Labeled samples per batch; rest are unlabeled.")
    optim.add_argument("-j", "--workers", type=int, default=None,
                       help="Data loader workers (default: 4).")
    optim.add_argument("--lr", type=float, default=None,
                       help="Max learning rate (default: 1e-3).")
    optim.add_argument('--lr_min', type=float, default=1e-5,
            help='Starting learning rate for warmup schedule')
    optim.add_argument("--initial-lr", type=float, default=None,
                       help="Starting LR for ramp-up (default: 0).")
    optim.add_argument("--weight-decay", type=float, default=None,
                       help="Weight decay (default: 1e-4).")
    optim.add_argument("--gpu-id", type=int, default=None, help="GPU id to use")


    # ----------------------------
    # Semi-supervised schedule
    # ----------------------------
    semi = parser.add_argument_group("semi-supervised")
    semi.add_argument("--warmup-epochs", type=int, default=None,
                      help="Train only on labeled data for first N epochs.")
    semi.add_argument("--conf-threshold", type=float, default=None,
                      help="Confidence threshold for teacher mask (default: 0.8).")

    # ----------------------------
    # Knowledge Distillation (Two Teachers)
    # ----------------------------
    kd = parser.add_argument_group("distillation")
    kd.add_argument("--teacher-ckpt", type=str, default=None,
                    help="Path to Teacher A checkpoint (.pth/.ckpt).")
    kd.add_argument("--kd-weight", type=float, default=None,
                    help="Weight on KD (consistency) loss.")
    kd.add_argument("--kd-temp", type=float, default=None,
                    help="Temperature for KD logits (default: 1.0).")
    kd.add_argument("--fuse-rule", type=str, default=None,
                    choices=["max", "noisy_or"],
                    help="How to fuse Teacher A/B prob maps (default: max).")

    # ----------------------------
    # Logging / I/O
    # ----------------------------
    io = parser.add_argument_group("logging/io")
    io.add_argument("--print-freq", "-p", type=int, default=None,
                    help="Log frequency (iters).")
    io.add_argument("--checkpoint-epochs", type=int, default=None,
                    help="Save checkpoint every N epochs (default: 5).")
    io.add_argument("--resume", type=str, default=None,
                    help="Path to resume student state.")
    io.add_argument("-e", "--evaluate", action="store_true",
                help="Only run evaluation (no training).")


    return parser

def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
