# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import random
import numpy as np
import torch
import _init_paths
import dataset
from dataload import dataload
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from models.pose_shufflenetv2 import get_pose_net

# import dataset
# import models

# from models.checkpoint import load_checkpoint
import mindspore
from mindspore.dataset import GeneratorDataset 
import mindspore.dataset
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='/home/neo/Documents/wkz/ADPD_ms_x2ms/experiments/adpd_mpii/shuffnetv2/hg8_shuffnetv2_ADPD.yml')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

def setup_seed(seed):
    mindspore.set_seed(seed)
    mindspore.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
        + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
                shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


def main():
    args = parse_args()
    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    setup_seed(cfg.SEED)

    model = get_pose_net(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        t=torch.load(cfg.TEST.MODEL_FILE,map_location=torch.device('cpu'))        
        temp=model.parameters_and_names()
        for key, parameter in model.parameters_and_names():                # 获取ms模型的参数名和数值
            if 'embedding_table' in key:                                    # 参数名中的embedding_table替换为weight
                key = key.replace('embedding_table', 'weight')
            elif 'gamma' in key:
                key = key.replace('gamma', 'weight')                        #// 参数名中的gamma替换为weight
            elif 'beta' in key:
                key = key.replace('beta', 'bias')                           #// 参数名中的beta替换为bias
            elif 'moving_mean' in key:
                key = key.replace('moving_mean', 'running_mean')
            elif 'moving_variance' in key:
                key = key.replace('moving_variance', 'running_var')
            elif 'weight' in key:
                temp = mindspore.Tensor(t.get(key).detach().numpy())
                if len(parameter.shape)>len(temp.shape):
                    temp = temp.unsqueeze(len(temp.shape))
                    parameter.set_data(temp)
                    num+=1
                    continue
                #依据key获取pytorch中相应参数的数值并赋给mindspore当前参数parameter，上面替换参数名就是为了get(key)的时候不会找不到
            temp=mindspore.Tensor(t.get(key).detach().numpy())
            parameter.set_data(temp)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        t=torch.load(model_state_file,map_location=torch.device('cpu'))        
        temp=model.parameters_and_names()
        for key, parameter in model.parameters_and_names():                # 获取ms模型的参数名和数值
            if 'embedding_table' in key:                                    # 参数名中的embedding_table替换为weight
                key = key.replace('embedding_table', 'weight')
            elif 'gamma' in key:
                key = key.replace('gamma', 'weight')                        #// 参数名中的gamma替换为weight
            elif 'beta' in key:
                key = key.replace('beta', 'bias')                           #// 参数名中的beta替换为bias
            elif 'moving_mean' in key:
                key = key.replace('moving_mean', 'running_mean')
            elif 'moving_variance' in key:
                key = key.replace('moving_variance', 'running_var')
            elif 'weight' in key:
                temp = mindspore.Tensor(t.get(key).detach().numpy())
                if len(parameter.shape)>len(temp.shape):
                    temp = temp.unsqueeze(len(temp.shape))
                    parameter.set_data(temp)
                    num+=1
                    continue
                #依据key获取pytorch中相应参数的数值并赋给mindspore当前参数parameter，上面替换参数名就是为了get(key)的时候不会找不到
            temp=mindspore.Tensor(t.get(key).detach().numpy())
            parameter.set_data(temp)

    # model = x2ms_nn.DataParallel(model, device_ids=cfg.GPUS)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )

    # Data loading code
    valid_dataset,valid_loader=dataload(cfg=cfg)
    
    # valid_loader=valid_loader.batch(cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS))
    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
