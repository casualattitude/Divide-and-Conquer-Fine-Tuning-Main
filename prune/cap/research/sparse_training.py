#!/usr/bin/env python3

""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
import json
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
import sys
sys.path.append('../sparseml')
# sparseml imports
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleSparsificationInfo, save_model, load_model, load_optimizer, load_epoch
# import utils
from aux_utils import is_update_epoch, get_current_sparsity, get_sparsity_info
from aux_utils.summary import update_summary

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
#parser.add_argument('--pretrained_weight',default='../weights/transimagenet_checkpoint-399.pth',type=str,help='Path to model checkpoint')
parser = argparse.ArgumentParser(description='Sparse training with timm script')
parser.add_argument('--pretrained_weight', default='../weights/transimagenet_checkpoint-399.pth', type=str, help='Path to model checkpoint')
# SparseML recipe
parser.add_argument('--sparseml-recipe', required=True, type=str,
                    help='YAML config file with the sparsification recipe')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='decay factor for model weights moving average (default: 0.9998)')

# Sparsification params
group = parser.add_argument_group('Sparsification params')
group.add_argument('--gs-loader', action='store_true',default='False',
                    help='whether to create loader for GradSampler (default: False)')
group.add_argument('-gb', '--gs-batch-size', type=int, default=16, metavar='N',
                    help='batch size of gs loader (default: None)')
group.add_argument('--gs-no-training', action='store_true',
                    help='Whether to use GradSampler loader in eval mode')
group.add_argument('--gs-no-aug', action='store_true',
                    help='Whether to remove augmentation from GradSampler')
group.add_argument('--gs-aa', action='store_true',
                    help='Whether to apply auto augmentation in GradSampler')
group.add_argument('--gs-distributed', action='store_true',
                    help='Whether to use GradSampler in non-distributed regime')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('--checkpoint-freq', type=int, default=-1, metavar='N',
                   help='checkpointing frequency (default: no saving epoch checkpoints)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--save-last', action='store_true', default=False,
                    help='Whether to save the last state of the model')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
group.add_argument('--log-sparsity', action='store_true', default=False,
                   help='whether to log sparsity on each pruning step')
group.add_argument('--log-param-histogram', action='store_true', default=False,
                   help='Log histogram of params (works only if log_wandb = True)')
group.add_argument('--timeout', type=int, default=1800,
                   help='Worker timeout')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # init distributed training 设置分布式训练
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method='env://', 
            timeout=timedelta(seconds=args.timeout)
        )
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    args.device = torch.device(args.device)
    device = args.device
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
    #设置随机种子
    utils.random_seed(args.seed, args.rank)
    #硬件加速用的
    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()
    #初始化输入通道数为3。
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    #根据参数创建模型，包括设置预训练、输入通道数、输出类别数、dropout等
    '''model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **args.model_kwargs,
    )'''
    checkpoint = torch.load(args.pretrained_weight)
    model_state_dict = checkpoint['model']
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    try:
        # 尝试加载状态字典
        model.load_state_dict(model_state_dict, strict=False)
        print("Successfully loaded model weights.")
    except RuntimeError as e:
        # 捕获加载状态字典时出现的错误
        print(f"Error in loading model weights: {e}")
        # 在这里可以处理意外的键或者缺失的键

    #默认None
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    #默认None
    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')
    #创建优化器
    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        # load model checkpoint
        load_model(args.resume, model, fix_data_parallel=True)
        if args.local_rank == 0:
            _logger.info(f'Loading model from checkpoint {args.resume}')
        # load optimizer
        if not args.no_resume_opt:
            if args.local_rank == 0:
                _logger.info(f'Loading optimizer from checkpoint {args.resume}')
            load_optimizer(args.resume, optimizer, map_location=args.device)
        if not args.no_resume_epoch:
            resume_epoch = load_epoch(args.resume, map_location=args.device) + 1
            if args.local_rank == 0:
                _logger.info(f'Starting training from {resume_epoch} epoch')
    #启用了指数移动平均模型EMA，则根据参数创建EMA对象
    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)

    # create the train and eval datasets#创建数据集
    if args.data and not args.data_dir:
        args.data_dir = args.data
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
    )

    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
    )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
    )

    # setup loss function#设置损失函数
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    # #为训练过程设置检查点保存器、输出目录和Wandb日志记录
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    decreasing  = True if 'loss' in eval_metric else False
    # set comparison
    if decreasing:
        is_better = lambda x, y: x < y 
    else:
        is_better = lambda x, y: x > y
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    #########################
    # Setup SparseML manager
    ############$############

    manager = ScheduledModifierManager.from_yaml(args.sparseml_recipe)
    print("manager.iter_modifiers()",manager.iter_modifiers())
    #print("manager.modifiers:",manager.modifiers)yaml
    #print("manager.metadata:", manager.metadata)None
    #print("manager is:",manager)
    manager_kwargs = {}
    # make separate OBS-loader
    if args.gs_loader: 
        loader_gs = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.gs_batch_size,
            is_training=not args.gs_no_training,
            no_aug=args.gs_no_aug,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            num_workers=args.workers,
            distributed=args.gs_distributed,
            auto_augment=args.aa if args.gs_aa else None,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem
        )

        def data_loader_builder(device=args.device, **kwargs):
            while True:
                for input, target in loader_gs:
                    input, target = input.to(device), target.to(device)
                    yield [input], {}, target

        manager_kwargs['grad_sampler'] =  {
            'data_loader_builder' : data_loader_builder,
            'loss_fn' : validate_loss_fn
        }
    #print("manager_kwargs",manager_kwargs)
    #对给定的模型和优化器进行修改，以适应特定的稀疏性算法，并返回一个包装后的优化器对象
    optimizer = manager.modify(
        model, 
        optimizer, 
        steps_per_epoch=len(loader_train), 
        epoch=start_epoch,
        #grad_sampler=manager_kwargs.get('grad_sampler', None),
        **manager_kwargs
    ) 
    #print("1")
    #print("manager.modifiers are:",manager.modifiers)
    # override timm scheduler#不再使用 timm 的学习率调度器，而是使用 SparseML 配方中的指令来管理学习率
    if any("LearningRate" in str(modifier) for modifier in manager.modifiers):
        lr_scheduler = None
        if manager.max_epochs:
            num_epochs = manager.max_epochs# 300
        print("args.local_rank:",args.local_rank)
        if args.local_rank == 0:
            _logger.info("Disabling timm LR scheduler, managing LR using SparseML recipe")
            _logger.info(f"Overriding max_epochs to {num_epochs} from SparseML recipe")

    if utils.is_primary(args) and lr_scheduler is not None:#不执行
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')
    #print("2")
    try:
        # 通过 dataset_train.set_epoch(epoch) 或 loader_train.sampler.set_epoch(epoch) 来设置训练数据集的当前 epoch。
        # 确保数据集在每个 epoch 都能以正确的顺序提供数据
        for epoch in range(start_epoch, num_epochs):#0-300
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            #print("3")
            train_metrics = train_one_epoch(#返回训练一个epoch的损失
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):#在分布式训练环境中处理批归一化（Batch Normalization）层的运行时均值和方差
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
            )

            # SparseML summary
            sparsity_info = ModuleSparsificationInfo(model)#计算模型的稀疏性信息，包括总参数数量、稀疏参数数量、剪枝情况、量化情况等，并将其转换为字典形式，方便后续处理和记录
            sparsity_summary = json.loads(str(sparsity_info))#总参数数量、稀疏参数数量、剪枝情况、量化情况
            #print("4")
             # get current mean sparsity
            scheduled_sparsity = get_current_sparsity(manager, epoch)#获取当前的平均稀疏度
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                # 对EMA模型进行分布式BatchNorm分布或广播，然后进行EMA模型的评估
                # 但如果当前处于稀疏阶段（scheduled_sparsity == 0.0），则不对EMA模型进行评估
                # do not evaluate EMA if in sparse stage
                if scheduled_sparsity == 0.0:#如果稀疏度为0.0，表示当前模型处于未稀疏化的阶段，即没有进行剪枝操作。在这种情况下，代码调用了 validate 函数，该函数用于在验证集上评估模型
                    ema_eval_metrics = validate(
                        model_ema.module,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics


            if args.local_rank == 0:#如果本地rank为0（一般指主进程），则进行一系列日志记录和保存操作
                if output_dir is not None:
                    lr_param = [param_group['lr'] for param_group in optimizer.param_groups]
                    lr_m = sum(lr_param) / len(lr_param)
                    # log param histogram
                    param_hist = {}
                    # 启用了参数直方图记录 args.log_param_histogram 且使用了 wandb 记录日志
                    # 则会为模型中的每个参数记录参数直方图。这些直方图存储在 param_hist 字典中
                    if args.log_param_histogram and args.log_wandb:
                        for param_name, param in model.named_parameters():
                            # strip module
                            module_key = 'module.'
                            if param_name.startswith(module_key):
                                param_name = param_name[len(module_key):]
                            param_nnz = param[param != 0.0].detach().cpu().numpy()
                            param_hist[param_name] = wandb.Histogram(param_nnz)
                    # get current lr
                    update_summary(#调用 update_summary 函数更新摘要信息，包括当前的训练指标、验证指标、文件名、学习率、稀疏度等信息
                        epoch,
                        train_metrics,
                        eval_metrics,
                        filename=f'{output_dir}/summary.csv',
                        write_header=(epoch == start_epoch),
                        log_wandb=args.log_wandb and has_wandb,
                        param_hist=param_hist,
                        lr=lr_m,
                        sparsity=scheduled_sparsity,
                        **sparsity_summary["params_summary"]
                    )
                #如果当前 epoch 能被 args.checkpoint_freq 整除且 args.checkpoint_freq 大于0，则调用 save_model 函数保存模型
                if epoch % args.checkpoint_freq == 0 and args.checkpoint_freq > 0:
                    save_model(
                        path=f'{output_dir}/{args.model}_epoch={epoch}.pth',
                        model=model, 
                        optimizer=optimizer, 
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )

                # log scheduled sparsity
                _logger.info(f'Scheduled sparsity: {scheduled_sparsity:.4f}')#记录当前的计划稀疏度信息到日志中
                #如果启用了稀疏度日志记录 args.log_sparsity 并且当前 epoch 是更新周期（由 is_update_epoch 函数确定）
                # 则获取模型的稀疏度信息，并将其存储为 JSON 文件，文件名中包含了当前 epoch 数量
                if args.log_sparsity and is_update_epoch(manager, epoch):
                    sparsity_info = json.loads(get_sparsity_info(model))
                    with open(f'{output_dir}/sparsity_distribution_epoch={epoch}.json', 'w') as outfile:
                        json.dump(sparsity_info, outfile)
                    # reset current best metric 重置当前最佳指标 best_metric 为 None，为下一个 epoch 的评估做准备
                    best_metric = None
                # save best checkpoint
                #首先检查当前的最佳指标是否为空或者当前评估指标是否比之前的最佳指标更好
                if best_metric is None or is_better(eval_metrics[eval_metric], best_metric):
                    best_metric = eval_metrics[eval_metric]#更新最佳指标和最佳的 epoch 数
                    best_epoch = epoch
                    save_model(#调用 save_model 函数保存当前模型的检查点。检查点的路径包括输出目录 output_dir、模型名称和稀疏度信息，以及当前 epoch 的信息
                        path=f'{output_dir}/{args.model}_sparsity={scheduled_sparsity:.2f}_best.pth',
                        model=model, 
                        optimizer=optimizer, 
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        use_zipfile_serialization_if_available=True,
                        include_modifiers=True
                    )
                    #通过日志记录新的最佳模型的信息，包括稀疏度、epoch 和准确度
                    _logger.info(f'New best model for sparsity {scheduled_sparsity:.2f} on epoch {epoch} with accuracy {best_metric:.4f}')

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass
    # optionally save last checkpoint
    # #保存最后一个检查点
    if args.save_last:
        save_model(
            path=f'{output_dir}/{args.model}_last.pth',
            model=model, 
            optimizer=optimizer, 
            loss_scaler=loss_scaler,
            epoch=epoch,
            use_zipfile_serialization_if_available=True,
            include_modifiers=True
        )
        _logger.info(f'Saved last checkpoint.')        

    # finalize manager
    manager.finalize(model)
    if args.local_rank == 0:
        if best_metric is not None:
            _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
        _logger.info('Training completed. Have a nice day!')
        if args.log_wandb and has_wandb:
            wandb.finish()


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device=torch.device('cuda'),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order#检查优化器是否支持二阶优化
    has_no_sync = hasattr(model, "no_sync")#检查模型是否有不需要同步的参数
    #记录训练过程中的时间、数据加载时间和损失值
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    #print("train one epoch")
    model.train()
    #初始化梯度累积步数
    accum_steps = args.grad_accum_steps
    #计算最后一个 batch 的累积步数
    last_accum_steps = len(loader) % accum_steps
    #计算每个 epoch 的更新步数
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    #初始化总更新步数
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    #获取最后一个需要累积的 batch 的索引
    last_batch_idx_to_accum = len(loader) - last_accum_steps
    #print("5")
    data_start_time = update_start_time = time.time()

    optimizer.zero_grad()#梯度置零
    update_sample_count = 0#记录更新的样本数量
    for batch_idx, (input, target) in enumerate(loader):
        #迭代数据加载器中的每个 batch
        last_batch = batch_idx == last_batch_idx#检查当前 batch 是否为最后一个 batch
        #判断是否需要进行参数更新。如果当前是最后一个 batch，或者当前 batch 是累积步数的倍数，就需要进行参数更新
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        #计算当前更新的索引
        update_idx = batch_idx // accum_steps
        #如果当前 batch 大于等于最后一个需要累积的 batch，则将累积步数设置为最后一个 batch 的累积步数
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps
       # print("6")
        if not args.prefetcher:#如果没有启用预取器
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:#如果使用了通道优先的内存格式
            input = input.contiguous(memory_format=torch.channels_last)
       # print("11")
        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))#更新数据加载时间统计
        #print("12")
        def _forward():
            with amp_autocast():#获取模型输出和损失
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
           # print("back1")
            if loss_scaler is not None:
                #print("backlossca")
                loss_scaler(#处理混合精度训练中损失缩放
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
               # print("backlossca2")
            else:
                #print("back2")
                _loss.backward(create_graph=second_order)
                if need_update:
                    #print("back3")
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                      #  print("back4")
                    optimizer.step()
       # print("13")
        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            #print("backward")
            _backward(loss)
       # print("14")
        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue
      #  print("15")
        num_updates += 1
        optimizer.zero_grad()#梯度置零，准备进行下一轮参数更新
        if model_ema is not None:#如果有模型指数移动平均
            model_ema.update(model)

        if args.synchronize_step and device.type == 'cuda':#需要同步步骤，并且设备类型为 CUDA
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now
      #  print("16")
        if update_idx % args.log_interval == 0:#如果当前更新的索引是日志间隔的倍数
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)#得到了所有参数组学习率的平均值

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)#对损失值进行全局聚合
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))#使用全局损失值更新 losses_m 的平均值
                update_sample_count *= args.world_size#调整更新的样本数

            if utils.is_primary(args):#如果当前是主进程
                _logger.info(#打印训练信息，包括当前 epoch、更新进度、损失、时间等
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:#默认不执行
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (#默认不执行
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)
       # print("7")
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)#学习率调度
        #print("8")
        update_sample_count = 0#重置样本数量统计
        data_start_time = time.time()
        # end for
       # print("9")
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
        #print("10")
    #print("end train one epoch")
    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1#获取数据加载器的长度，减去 1 是为了得到最后一个索引
    with torch.no_grad():#进入不计算梯度模式
        for batch_idx, (input, target) in enumerate(loader):#遍历数据加载器中的每个 batch
            last_batch = batch_idx == last_idx#last_batch判断当前 batch 是否为最后一个 batch
            if not args.prefetcher:#如果没有启用预取器
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:#使用了通道优先的内存格式
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():#自动混合精度
                output = model(input)#模型前向传播，得到输出
                if isinstance(output, (tuple, list)):#如果输出是元组或列表类型
                    output = output[0]#取出第一个元素

                # augmentation reduction
                reduce_factor = args.tta#测试时增强
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)#计算损失值
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))#计算 Top-1 和 Top-5 准确率

            if args.distributed:#如果启用了分布式训练
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)#对损失值进行分布式同步
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))#更新损失统计
            top1_m.update(acc1.item(), output.size(0))#更新 Top-1 准确率统计
            top5_m.update(acc5.item(), output.size(0))#更新 Top-5 准确率统计

            batch_time_m.update(time.time() - end)#更新每个 batch 的时间统计
            end = time.time()#更新时间
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):#如果当前是主进程，并且是最后一个 batch，或者当前 batch 是日志间隔的倍数
                log_name = 'Test' + log_suffix
                _logger.info(#打印验证信息，包括时间、损失、准确率等
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])#构建一个有序字典，包含损失和准确率的平均值

    return metrics#返回验证指标


if __name__ == '__main__':
    main()
