# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
# import numpy.ma as ma
from tqdm import tqdm

import torch
# import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils import AverageMeter
from utils import get_confusion_matrix
from utils import adjust_learning_rate
from distributed import is_distributed
from utils import get_world_size, get_rank


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, model, writer_dict, criterion, args):  # writer_dict
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    m_oa = AverageMeter()
    m_iou = AverageMeter()
    fw_iou = AverageMeter()

    m_oa_ocr = AverageMeter()
    m_iou_ocr = AverageMeter()
    fw_iou_ocr = AverageMeter()

    tic = time.time()
    cur_iters = epoch * epoch_iters

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.long().cuda(args.gpu, non_blocking=True)

        size = labels.size()
        # images = images.to(device)
        # labels = labels.long().to(device)

        preds = model(images)
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        for i, x in enumerate(preds):
            x = F.interpolate(
                input=x, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            confusion_matrix[..., i] = get_confusion_matrix(
                labels,
                x,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL
            )

            pos = confusion_matrix[..., i].sum(1)
            res = confusion_matrix[..., i].sum(0)
            tp = np.diag(confusion_matrix[..., i])
            oa = np.sum(tp) / (np.sum(confusion_matrix[..., i]) + 1e-7)
            IoU_array = tp / np.maximum(1.0, pos + res - tp)
            mean_IoU = IoU_array.mean()
            freq = np.sum(confusion_matrix[..., i], axis=1) / (np.sum(confusion_matrix[..., i]) + 1e-7)
            iu = np.diag(confusion_matrix[..., i]) / (
                    np.sum(confusion_matrix[..., i], axis=1) + np.sum(confusion_matrix[..., i], axis=0) -
                    np.diag(confusion_matrix[..., i]) + 1e-7)
            FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
            if i == 0:
                m_oa.update(oa, images.size(0))
                m_iou.update(mean_IoU, images.size(0))
                fw_iou.update(FWIoU, images.size(0))
            else:
                m_oa_ocr.update(oa, images.size(0))
                m_iou_ocr.update(mean_IoU, images.size(0))
                fw_iou_ocr.update(FWIoU, images.size(0))
        losses = criterion(preds, labels, config)
        loss = losses.mean()
        if is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        # update average loss
        ave_loss.update(reduced_loss.item())
        # ave_loss.update(losses.item(), images.size(0))

        # compute gradient and do SGD step
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if (i_iter + 1) % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'HRNet Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f},lr: {:.6f}, Loss: {:.6f}, Mean_IoU: {:.4f}, OA: {:.4f}, FWIoU: {:.4f}' \
                .format(epoch, num_epoch, i_iter + 1, epoch_iters, batch_time.average(), lr,
                        print_loss, m_iou.average(), m_oa.average(), fw_iou.average())

            msg_ocr = 'OCR Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f},lr: {:.6f}, Loss: {:.6f}, Mean_IoU: {:.4f}, OA: {:.4f}, FWIoU: {:.4f}' \
                .format(epoch, num_epoch, i_iter + 1, epoch_iters, batch_time.average(), lr,
                        print_loss, m_iou_ocr.average(), m_oa_ocr.average(), fw_iou_ocr.average())

            logging.info(msg)
            logging.info(msg_ocr)
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict, criterion, args):  # writer_dict
    rank = get_rank()
    # world_size = get_world_size()
    model.eval()
    # batch_time = AverageMeter()
    m_oa = AverageMeter()
    m_iou = AverageMeter()
    fw_iou = AverageMeter()

    m_oa_ocr = AverageMeter()
    m_iou_ocr = AverageMeter()
    fw_iou_ocr = AverageMeter()

    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))

    with torch.no_grad():
        # end = time.time()
        for _, batch in enumerate(testloader):
            # image, label = batch
            images, labels, _, _ = batch

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.long().cuda(args.gpu, non_blocking=True)

            size = labels.size()
            # images = images.to(device)
            # labels = labels.long().to(device)
            pred = model(images)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] = get_confusion_matrix(
                    labels,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

                pos = confusion_matrix[..., i].sum(1)
                res = confusion_matrix[..., i].sum(0)
                tp = np.diag(confusion_matrix[..., i])
                oa = np.sum(tp) / (np.sum(confusion_matrix[..., i]) + 1e-7)
                IoU_array = tp / np.maximum(1.0, pos + res - tp)
                mean_IoU = IoU_array.mean()
                freq = np.sum(confusion_matrix[..., i], axis=1) / (np.sum(confusion_matrix[..., i]) + 1e-7)
                iu = np.diag(confusion_matrix[..., i]) / (
                        np.sum(confusion_matrix[..., i], axis=1) + np.sum(confusion_matrix[..., i], axis=0) -
                        np.diag(confusion_matrix[..., i]) + 1e-7)
                FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

                if i == 0:
                    m_oa.update(oa, images.size(0))
                    m_iou.update(mean_IoU, images.size(0))
                    fw_iou.update(FWIoU, images.size(0))
                else:
                    m_oa_ocr.update(oa, images.size(0))
                    m_iou_ocr.update(mean_IoU, images.size(0))
                    fw_iou_ocr.update(FWIoU, images.size(0))

            OA = [m_oa.average(), m_oa_ocr.average()]
            mean_IoU = [m_iou.average(), m_iou_ocr.average()]
            FWIoU = [fw_iou.average(), fw_iou_ocr.average()]
            # batch_time.update(time.time() - end)
            # end = time.time()
    # confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    # reduced_confusion_matrix = reduce_tensor(confusion_matrix)
    # confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    # IoU_array = []
    # for i in range(nums):
    #     pos = confusion_matrix[..., i].sum(1)
    #     res = confusion_matrix[..., i].sum(0)
    #     tp = np.diag(confusion_matrix[..., i])
    #     freq = np.sum(confusion_matrix[..., i], axis=1) / np.sum(confusion_matrix[..., i])
    #     iu = np.diag(confusion_matrix[..., i]) / (
    #             np.sum(confusion_matrix[..., i], axis=1) + np.sum(confusion_matrix[..., i], axis=0) -
    #             np.diag(confusion_matrix[..., i]))
    #
    #     IoU_array.append((tp / np.maximum(1.0, pos + res - tp)))
    #     OA.append(np.sum(tp) / np.sum(confusion_matrix[..., i]))
    #     mean_IoU.append(IoU_array[i].mean())
    #     FWIoU.append((freq[freq > 0] * iu[freq > 0]).sum())

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        # writer.add_scalar('valid_mIoU', mean_IoU.average(), global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return mean_IoU, OA, FWIoU, rank
