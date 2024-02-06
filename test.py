import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from utils.tools import set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.corruption_datasets import corruptions_datasets
import utils,methods
from data.imagenet_ar import ImageNetdata

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    args = parser.parse_args()
    args.method = utils.config2args(argparse.Namespace(), args.method_config)
    args.stream = utils.config2args(argparse.Namespace(), args.stream_config)
    archString=args.arch.replace("/","")
    print("archString:",archString)
    args.exp=f"{archString}_{args.stream.name}_{args.method.name}" +f"_{args.seed}"+f"_S{args.stream.severities[0]}"
    set_random_seed(args.seed)

    if args.method.method=='ours':
        args.ours=True
        args.method.chooseThreshold = args.threshold
        args.method.queue_size = args.queue_size
        args.exp+=f"_{args.threshold}_{args.queue_size}"
        if args.prompt!=-1:
            args.exp+=f"_{args.prompt}"

    logger=utils.set_logger(args.result_dir, args.exp)
    utils.log_args(args,logger)

    assert args.gpu is not None
    main_worker(args.gpu, args,logger)


def main_worker(gpu, args,logger):
    args.gpu = gpu
    set_random_seed(args.seed)
    logger.info(f"Use GPU: {args.gpu} for training")

    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    elif args.test_sets in corruptions_datasets:
        classnames=eval("{}_classes".format(args.test_sets))
    elif args.test_sets in ImageNetdata:
        classnames=eval("{}".format(args.test_sets))
    else:
        classnames = imagenet_classes
    logger.info(f"classnames: {classnames}")

    model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init,multi=args.ours)

    if not hasattr(methods, args.method.method): logger.error(f"Method {args.method.method} is not found in TTA method set.")
    ##写到这里来了
    net = getattr(methods, args.method.method).setup(model, args,logger,args.test_sets)
    #这里传入的参数为:(相关模型,参数列表,打印的logger,测试集的名称)
    
    net.recordResult()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=400, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stream-config',default="./configs/stream/CIFAR10.yaml",  type=str, help='path of stream config file')#
    parser.add_argument('--method-config',default="./configs/methods/tpt.yaml"  ,type=str, help="path of method config file")#
    parser.add_argument('--result-dir',  default="./results", type=str, help="path storing results")
    parser.add_argument('--ours',action='store_true',help='Select to Update C1')
    parser.add_argument('--tptContinual',action='store_true',help='Select to Update tpt continually')
    parser.add_argument('--threshold',type=float, default=0.7,help='Threshold to choose sample')
    parser.add_argument('--queue_size',type=int, default=64,help='Buffer size')
    parser.add_argument('--prompt',type=int, default=-1,help='Choose Prompt')
    main()