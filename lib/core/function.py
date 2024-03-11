# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Written by Feng Zhang & Hong Hu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mindspore
import mindspore.context as context
import x2ms_adapter
from x2ms_adapter.core.context import x2ms_context
from x2ms_adapter.torch_api.optimizers import optim_register
from x2ms_adapter.core.cell_wrapper import WithLossCell
import mindspore
import x2ms_adapter
from x2ms_adapter.auto_static import auto_static

# mindspore.set_context(mode=mindspore.GRAPH_MODE,save_graphs=2, save_graphs_path='./ir')

import time
import logging
import os

import numpy as np

import mindspore.dataset.vision as vision
from core.evaluation import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    x2ms_adapter.x2ms_train(model)

    end = time.time()
    # (TRANSPLANT ADVICE) 1.Move the calculation logic of loss to this function and 
    # let it return loss.
    # The implementation function must comply with the graph mode syntax standard.
    def loss_func():
        return None
    
    # (TRANSPLANT ADVICE) 3.Add required variables to the arguments of construct function.
    # Only some basic types, Tensor and Parameter, are supported.
    def construct(self, input):
        outputs = model(input)

        # (TRANSPLANT ADVICE) 2.Pass in the variables required to calculate the loss for the loss function here
        loss = loss_func()

        return loss, outputs
    
    wrapped_model = WithLossCell(construct = construct, key = 0)
    wrapped_model = x2ms_adapter.graph_train_one_step_cell(wrapped_model, optim_register.get_instance())
    wrapped_model.compile_cache.clear()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # (TRANSPLANT ADVICE) 4. Add required variables to the wrapped_model's arguments.
        loss, outputs = wrapped_model.call_construct(input)

        target = target
        target_weight = target_weight

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
            output = outputs[-1]
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        x2ms_adapter.nn_cell.zero_grad(optimizer)
        # loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(x2ms_adapter.tensor_api.item(loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))

        _, avg_acc, cnt, pred = accuracy(x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(output)),
                                         x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(target)))
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=x2ms_adapter.tensor_api.x2ms_size(input, 0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def fpd_train(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
              output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA

    # s_model switch to train mode and t_model switch to evaluate mode
    x2ms_adapter.x2ms_train(model)
    x2ms_adapter.x2ms_eval(tmodel)

    end = time.time()
    # (TRANSPLANT ADVICE) 1.Move the calculation logic of loss to this function and 
    # let it return loss.
    # The implementation function must comply with the graph mode syntax standard.
    def loss_func():
        return None
    
    # (TRANSPLANT ADVICE) 3.Add required variables to the arguments of construct function.
    # Only some basic types, Tensor and Parameter, are supported.
    def construct(self, input):
        outputs = model(input)

        # (TRANSPLANT ADVICE) 2.Pass in the variables required to calculate the loss for the loss function here
        loss = loss_func()

        return loss, outputs
    
    wrapped_model = WithLossCell(construct = construct, key = 1)
    wrapped_model = x2ms_adapter.graph_train_one_step_cell(wrapped_model, optim_register.get_instance())
    wrapped_model.compile_cache.clear()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # (TRANSPLANT ADVICE) 4. Add required variables to the wrapped_model's arguments.
        loss, outputs = wrapped_model.call_construct(input)
        toutput = tmodel(input)

        if isinstance(toutput, list):
            toutput = toutput[-1]

        target = target
        target_weight = target_weight

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], toutput, target_weight, target=target)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target, target_weight)
                kd_pose_loss += kd_pose_criterion(output, toutput, target_weight, target=target)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput, target_weight, target=target)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        x2ms_adapter.nn_cell.zero_grad(optimizer)
        # loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(x2ms_adapter.tensor_api.item(pose_loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))
        kd_pose_losses.update(x2ms_adapter.tensor_api.item(kd_pose_loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))
        losses.update(x2ms_adapter.tensor_api.item(loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))

        _, avg_acc, cnt, pred = accuracy(x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(output)),
                                         x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(target)))
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=x2ms_adapter.tensor_api.x2ms_size(input, 0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def adpd_train(config, train_loader, model, tmodel, weight, pose_criterion, kd_pose_criterion,
               optimizer, epoch, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()

    BiKD_losses = AverageMeter()
    KAD_losses = AverageMeter()
    NAD_losses = AverageMeter()

    kd_weight_alpha = config.KD.ALPHA

    # s_model and weight switch to train mode and t_model switch to evaluate mode
    x2ms_adapter.x2ms_train(model)
    x2ms_adapter.x2ms_train(weight)
    x2ms_adapter.x2ms_eval(tmodel)

    end = time.time()
    # (TRANSPLANT ADVICE) 1.Move the calculation logic of loss to this function and 
    # let it return loss.
    # The implementation function must comply with the graph mode syntax standard.
    def loss_func():
        return None
    
    # (TRANSPLANT ADVICE) 3.Add required variables to the arguments of construct function.
    # Only some basic types, Tensor and Parameter, are supported.
    def construct(self, input):
        outputs = model(input)

        # (TRANSPLANT ADVICE) 2.Pass in the variables required to calculate the loss for the loss function here
        loss = loss_func()

        return loss, outputs
    
    wrapped_model = WithLossCell(construct = construct, key = 2)
    wrapped_model = x2ms_adapter.graph_train_one_step_cell(wrapped_model, optim_register.get_instance())
    wrapped_model.compile_cache.clear()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # (TRANSPLANT ADVICE) 4. Add required variables to the wrapped_model's arguments.
        loss, outputs = wrapped_model.call_construct(input)
        toutput = tmodel(input)
        
        if isinstance(toutput, list):
            toutput = toutput[-1]
        
        target = target
        target_weight = target_weight
        
        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[-1], target, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[-1], toutput, target, target_weight)

            weights = weight(kd_pose_loss['BiKD'], kd_pose_loss['KAD'], kd_pose_loss['NAD'],
                             outputs[-1], toutput)
            
            alpha = weights[:, 0]; beta = weights[:, 1]; theta = weights[:, 2]

            kd_loss = x2ms_adapter.tensor_api.x2ms_mean((alpha * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['bi'], -1) - x2ms_adapter.log(alpha)))
            kd_loss += x2ms_adapter.tensor_api.x2ms_mean((beta * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['p'], -1) - x2ms_adapter.log(beta)))
            kd_loss += x2ms_adapter.tensor_api.x2ms_mean((theta * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['b'], -1) - x2ms_adapter.log(theta)))
            
            for output in outputs[:-1]:
                pose_loss += pose_criterion(output, target, target_weight)
                kd_pose_loss = kd_pose_criterion(output, toutput, target, target_weight)

                kd_loss = x2ms_adapter.tensor_api.x2ms_mean((alpha * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['BiKD'], -1) - x2ms_adapter.log(alpha)))
                kd_loss += x2ms_adapter.tensor_api.x2ms_mean((beta * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['KAD'], -1) - x2ms_adapter.log(beta)))
                kd_loss += x2ms_adapter.tensor_api.x2ms_mean((theta * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['NAD'], -1) - x2ms_adapter.log(theta)))

            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput, target, target_weight)

            weights = weight(kd_pose_loss['BiKD'], kd_pose_loss['KAD'], kd_pose_loss['NAD'],
                             outputs[-1], toutput)
            
            alpha = weights[:, 0]; beta = weights[:, 1]; theta = weights[:, 2]

            kd_loss = x2ms_adapter.tensor_api.x2ms_mean((alpha * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['BiKD'], -1) - x2ms_adapter.log(alpha)))
            kd_loss += x2ms_adapter.tensor_api.x2ms_mean((beta * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['KAD'], -1) - x2ms_adapter.log(beta)))
            kd_loss += x2ms_adapter.tensor_api.x2ms_mean((theta * x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['NAD'], -1) - x2ms_adapter.log(theta)))

            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_loss
        
        x2ms_adapter.nn_cell.zero_grad(optimizer)
        # loss.backward()
        optimizer.step()

        pose_losses.update(x2ms_adapter.tensor_api.item(pose_loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))
        kd_pose_losses.update(x2ms_adapter.tensor_api.item(kd_loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))
        losses.update(x2ms_adapter.tensor_api.item(loss), x2ms_adapter.tensor_api.x2ms_size(input, 0))

        BiKD_losses.update(x2ms_adapter.tensor_api.item((x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['BiKD']))), x2ms_adapter.tensor_api.x2ms_size(input, 0))
        KAD_losses.update(x2ms_adapter.tensor_api.item((x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['KAD']))), x2ms_adapter.tensor_api.x2ms_size(input, 0))
        NAD_losses.update(x2ms_adapter.tensor_api.item((x2ms_adapter.tensor_api.x2ms_mean(kd_pose_loss['NAD']))), x2ms_adapter.tensor_api.x2ms_size(input, 0))

        _, avg_acc, cnt, pred = accuracy(x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(output)),
                                         x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.detach(target)))
        acc.update(avg_acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Alpha:[{3:.3f}] Beta: [{4:.3f}] Theta: [{5:.3f}]\t' \
                  'BiKD {bi_loss.val:.5f} ({bi_loss.avg:.5f}) ' \
                  'KAD {k_loss.val:.5f} ({k_loss.avg:.5f}) ' \
                  'NAD {n_loss.val:.5f} ({n_loss.avg:.5f}) ' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f}) ' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), 
                      x2ms_adapter.tensor_api.item((x2ms_adapter.tensor_api.x2ms_mean(alpha))), x2ms_adapter.tensor_api.item((x2ms_adapter.tensor_api.x2ms_mean(beta))), x2ms_adapter.tensor_api.item((x2ms_adapter.tensor_api.x2ms_mean(theta))),
                      bi_loss=BiKD_losses, k_loss=KAD_losses, n_loss=NAD_losses,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.set_train(False)

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    end = time.time()
    for i,(input, target, target_weight, meta) in enumerate(val_loader):
        # compute output
        input=mindspore.tensor(input.numpy())
        target=mindspore.tensor(target.numpy())
        target_weight=mindspore.tensor(target_weight.numpy())
        meta['imgnum']=mindspore.tensor(meta['imgnum'].numpy())
        meta['joints']=mindspore.tensor(meta['joints'].numpy())
        meta['joints_vis']=mindspore.tensor(meta['joints_vis'].numpy())
        meta['center']=mindspore.tensor(meta['center'].numpy())
        meta['scale']=mindspore.tensor(meta['scale'].numpy())
        meta['rotation']=mindspore.tensor(meta['rotation'].numpy())
        meta['score']=mindspore.tensor(meta['score'].numpy())
        outputs = model(input)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input.asnumpy(), 3).copy()
            input_flipped = mindspore.Tensor(input_flipped)
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            t = flip_back(output_flipped.asnumpy(),val_dataset.flip_pairs)#return回来值就变了！！！
            output_flipped = mindspore.Tensor(t)


            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                        output_flipped[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        target = target
        target_weight = target_weight

        loss = criterion(output, target, target_weight)

        num_images = input.shape[0]
        # measure accuracy and record loss
        losses.update(x2ms_adapter.tensor_api.item(loss), num_images)
        _, avg_acc, cnt, pred = accuracy(output.asnumpy(),
                                         target.asnumpy())

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()

        preds, maxvals = get_final_preds(
            config, output.asnumpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        idx += num_images

        if i % config.PRINT_FREQ == 0:
            msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses, acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

    name_values, perf_indicator = val_dataset.evaluate(
        config, all_preds, output_dir, all_boxes, image_path,
        filenames, imgnums
    )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['& {:.3f}'.format(value) for value in values]) +
         ' &'
    )

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
        self.avg = self.sum / self.count if self.count != 0 else 0