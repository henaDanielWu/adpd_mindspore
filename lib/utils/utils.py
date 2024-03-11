from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path
from x2ms_adapter.torch_api.optimizers import optim_register
import mindspore
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
from x2ms_adapter.auto_static import auto_static

def create_logger(cfg, cfg_name, phase='train'):
    root_ouptput_dir = Path(cfg.OUTPUT_DIR)

    if not root_ouptput_dir.exists():
        print('=> creating {}'.format(root_ouptput_dir))
        root_ouptput_dir.mkdir()
    
    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if cfg.LOG_ABLATION:
        final_output_dir = root_ouptput_dir / dataset / model /cfg.LOG_ABLATION / cfg_name
    else:
        final_output_dir = root_ouptput_dir / dataset /model / cfg_name

    print('=> {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = phase + '_running_' + time_str + '.log'
    running_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(running_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = final_output_dir / 'tensorboard_log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_optimizer(cfg, model, weight=None):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim_register.sgd(
            x2ms_adapter.parameters(model),
            lr = cfg.TRAIN.LR,
            momentum=cfg.TRIAN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        if weight is None:
            optimizer = optim_register.adam(
                x2ms_adapter.parameters(model),
                lr=cfg.TRAIN.LR
            )
        else:
            optimizer = optim_register.adam(
                [{'params': x2ms_adapter.parameters(model), 'lr': cfg.TRAIN.LR},
                 {'params': x2ms_adapter.parameters(weight), 'lr': cfg.WEIGHT.LR}]
            )
    elif cfg.TRAIN.OPTIMIZER == 'rms':
        optimizer = optim_register.rmsprop(
            x2ms_adapter.parameters(model),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamwdecay': # optimizer for transformer
        params = []
        parameter_groups = {}
        # print(self.paramwise_cfg)
        num_layers = 12 + 2
        layer_decay_rate = 0.75
        print("Build LayerDecayOptimizerConstructor %f - %d" % (layer_decay_rate, num_layers))
        weight_decay = cfg.TRAIN.WD
        base_lr = cfg.TRAIN.LR

        for name, param in x2ms_adapter.named_parameters(model.module):
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'pos_embed' in name:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                # scale = layer_decay_rate ** (num_layers - layer_id - 1)
                if (num_layers - layer_id - 1) == 0:
                    scale = 2.0
                else:
                    scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * base_lr, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        params.extend(parameter_groups.values())
        optimizer = optim_register.adamw(
                params,
                lr=cfg.TRAIN.LR
            )
        
    return optimizer

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("blocks"):
        layer_id = int(x2ms_adapter.tensor_api.split(var_name, '.')[1])
        return layer_id + 1
    elif var_name.startswith("decoder_blocks"):
        layer_id = int(x2ms_adapter.tensor_api.split(var_name, '.')[1])
        return layer_id + 1 + 12
    else:
        return num_max_layer - 1

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    x2ms_adapter.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        x2ms_adapter.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in x2ms_adapter.parameters(module):
                    params += x2ms_adapter.tensor_api.x2ms_size(x2ms_adapter.tensor_api.view(param_, -1), 0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = x2ms_adapter.tensor_api.item((
                    x2ms_adapter.prod(
                        x2ms_adapter.LongTensor(list(x2ms_adapter.tensor_api.x2ms_size(module.weight.data)))) *
                    x2ms_adapter.prod(
                        x2ms_adapter.LongTensor(list(x2ms_adapter.tensor_api.x2ms_size(output))[2:]))))
            elif isinstance(module, x2ms_nn.Linear):
                flops = x2ms_adapter.tensor_api.item((x2ms_adapter.prod(x2ms_adapter.LongTensor(list(x2ms_adapter.tensor_api.x2ms_size(output)))) \
                         * x2ms_adapter.tensor_api.x2ms_size(input[0], 1)))

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(x2ms_adapter.tensor_api.x2ms_size(input[0])),
                    output_size=list(x2ms_adapter.tensor_api.x2ms_size(output)),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, x2ms_nn.ModuleList) \
           and not isinstance(module, x2ms_nn.Sequential) \
           and module != model:
            hooks.append(x2ms_adapter.nn_cell.register_forward_hook(module, hook))
    
    x2ms_adapter.x2ms_eval(model)
    x2ms_adapter.nn_cell.apply(model, add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

def load_checkpoint(checkpoint, model, 
                    optimizer=None, model_info=None, 
                    strict=True, filter_keys='not_filter'):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if x2ms_adapter.is_cuda_available():
        checkpoint = x2ms_adapter.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = x2ms_adapter.load(checkpoint, map_location=lambda storage, loc: storage)
    
    if len(checkpoint) < 10 and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
        if optimizer and 'optimizer' in checkpoint:
            x2ms_adapter.load_state_dict(optimizer, checkpoint['optimizer'])

    if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':  
        # the data parallel layer will add 'module' before each layer name    
        # load parallel model from checkpoint.pth.tar
        # remove score keys and remove module
        state_dict = {}
        for k, v in checkpoint.items():
            if filter_keys not in k:
                state_dict[str.replace(k, 'module.', '')] = v
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
        x2ms_adapter.load_state_dict(model, state_dict,strict=strict)
        
    print("=> loading {} model state_dict end! ".format(model_info))    
    return checkpoint

def save_yaml_file(cfg_name, cfg, final_output_dir):
    """
    :param cfg_name: config name
    :param cfg: config data
    :param final_output_dir: save file path name
    :return:
    """
    cfg_name = x2ms_adapter.tensor_api.split(os.path.basename(cfg_name), '.')[0]
    yaml_out_file = os.path.join(final_output_dir, cfg_name+'.yaml')
    print('=> creating {}'.format(yaml_out_file))
    with open(yaml_out_file, 'w') as outfile:
        outfile.write(cfg.dump())