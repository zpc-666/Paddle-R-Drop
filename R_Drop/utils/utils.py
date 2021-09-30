# coding=utf-8
from __future__ import division, print_function
import paddle
import logging
import os
import sys
sys.path.append("../")
from models import VisionTransformer, CONFIGS
from configs import parser_args
import numpy as np
import random
from datetime import timedelta
# 第1处改动 导入分布式训练所需的包
import paddle.distributed as dist

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model, optimizer, global_step, is_best=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    if not is_best:
        model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdparams" % args.name)
        paddle.save(model_to_save.state_dict(), model_checkpoint)
        paddle.save({'opt':optimizer.state_dict(), 'global_step':global_step}, os.path.join(args.output_dir, "%s_checkpoint.pdopt" % args.name))
    else:
        model_checkpoint = os.path.join(args.output_dir, "%s_best.pdparams" % args.name)
        paddle.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def resume(args, model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, "%s_checkpoint.pdparams" % args.name)
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, "%s_checkpoint.pdopt" % args.name)
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict['opt'])

            iter = opti_state_dict['global_step']
            iter = int(iter)
            return iter
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')

def setup():

    args = parser_args()
    
    device = paddle.device.get_device()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s, 16-bits training: %s" %
                   (args.device, args.n_gpu, args.fp16))

    # Set seed
    set_seed(args.seed)

    # Model & Tokenizer Setup
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "imagenet":
        num_classes=1000
    elif args.dataset == "cifar100":
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,alpha=args.alpha)
    model.load_from(np.load(args.pretrained_dir))
    num_params = count_parameters(model)

    if args.n_gpu>1:
        # 第2处改动，初始化并行环境
        dist.init_parallel_env()
        # 第3处改动，增加paddle.DataParallel封装
        model = paddle.DataParallel(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    #print(num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    return params/1000000


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)