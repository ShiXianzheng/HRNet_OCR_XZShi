import argparse
import logging
import os
# import shutil
import timeit
import warnings
import torch
import pprint
import _init_paths

from default import _C as config
from default import update_config
from models import MODEL_EXTRAS

import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import seg_hrnet_ocr
from modelsummary import get_model_summary
from data_loader_MultiScale import FUSARMapV2_MultiScale
from torch.utils.data.distributed import DistributedSampler
from criterion import CrossEntropy, OhemCrossEntropy
# from utils_origin import FullModel
from function_Distributed_hrnet import train, validate
from tensorboardX import SummaryWriter
import time
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist

os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5, 6, 7"

# python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
# python train_Distributed_inc4.py --cfg  --multiprocessing-distributed
# python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg
#  python3 -m torch.distributed.launch train_Distributed.py --cfg experimentals_yaml/hrnet_w18_train_flip_512_HB.yaml --multiprocessing-distributed


def parse_args():

    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument("--local_rank", type=int, default=4)

    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    # loading workers for sampler
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


# class FullModel(nn.Module):
#     """
#   Distribute the loss on multi-gpu to reduce
#   the memory cost in the main gpu.
#   You can check the following discussion.
#   https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
#   """
#
#     def __init__(self, model, loss):
#         super(FullModel, self).__init__()
#         self.model = model
#         self.loss = loss
#
#     def forward(self, inputs, labels):
#         outputs = self.model(inputs)
#         loss = self.loss(outputs, labels)
#         return torch.unsqueeze(loss, 0), outputs


def main():
    args = parse_args()
    # print(args)
    # print(config)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # distributed training
    # distributed 相较于 DataParallel，速度更快，效率更高
    # world size: 代表全局进程的个数， GPU数量
    # rank: 表示进程序号，用于进程间通讯，表征进程优先级
    # local rank: 进程内，GPU编号
    if args.dist_url == "env://" and args.world_size == -1:  # 使用 init_process_group 初始化进程组
        args.world_size = int(os.environ["WORLD_SIZE"])
    print(args.world_size)
    # 分布式训练判断
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 返回可用的GPU数量， node表示主机
    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)

    if args.multiprocessing_distributed:  # DDP？
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size  # 多机多卡，进程总数
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    args.gpu = gpu

    # 可利用的GPU
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')  # func
    # 将信息打印到控制台
    # logger.info(pprint.pformat(args))  # pprint可打印任意数据结构
    # logger.info(config)  # default

    # tb_log_dir
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    """
    pytorch中pytorch.cuda用于设置和运行CUDA操作，它会跟踪当前选定的GPU，
    并且您分配的所有CUDA张量将默认在该设备上创建。
    所选设备可以使用 torch.cuda.device 环境管理器进行更改。
    代码torch.cuda.is_available()判断当前环境是否可以使用GPU
    使用torch.device()创建一个torch.device对象，例如torch.device('cuda')或使用torch.device('cuda:0')指定GPU，
    该对象可以将张量移动至GPU上
    
    单机使用多个GPU有两种方式，torch.nn.DataParallel()与torch.nn.parallel.DistributedDataParallel 
    其中torch.nn.DataParallel()只能实现在单机多卡中进行分布式训练，
    而torch.nn.parallel.DistributedDataParallel则是新方法，在单机多卡和多机多卡都可以训练。
    官方建议使用最新的torch.nn.parallel.DistributedDataParallel，
    因为即使在单机多卡上，新的方法在效率上也要比旧的表现好

    """
    # cudnn related setting
    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    # 分布式训练进程组初始化
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # build model, create an instance of Neural Network, and init the weights from file
    # seg_hrnet.py  .get_seg_model
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # load pretrained

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:  # 分布式训练
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)  # 设置可用GPU
            model.cuda(args.gpu)
            # 在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。
            # 调用model.cuda()，可以将模型加载到GPU上去。这种方法不被提倡，而建议使用model.to(device)
            # 的方式，这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # workers将负责的batch加载进RAM
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # 创建并行模型
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:  # pipeline训练
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:  # DataParallel
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # if args.local_rank == 0:
    #     # provide the summary of model
    #     dump_input = torch.rand((1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]))
    #     # [batch, channel, height, width]
    #     logger.info(get_model_summary(model.to(args.gpu), dump_input.to(args.gpu)))

    # copy model file
    # this_dir = os.path.dirname(__file__)
    # models_dst_dir = os.path.join(final_output_dir, 'models')
    # if os.path.exists(models_dst_dir):
    #     shutil.rmtree(models_dst_dir)  # 删除目录及文件
    # shutil.copytree(os.path.join(this_dir, 'models'), models_dst_dir)  # 复制

    # torch.distributed, 再调整一下代码的分布式训练格式
    # if distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # class_weights = torch.FloatTensor([1., 1.]).cuda()
    class_weights = None

    # # criterion - loss, class weight, ignore label
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=class_weights).cuda(args.gpu)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=class_weights).cuda(args.gpu)

    # compile the model
    # model = FullModel(model, criterion)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)  # 数据通信，损失计算

    # torch.cuda.set_device(args.gpu)
    # model = model.cuda(args.gpu)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR},
                      {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')
    # if config.TRAIN.OPTIMIZER == 'sgd':
    #     optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()),
    #                                   'lr': config.TRAIN.LR}],
    #                                 lr=config.TRAIN.LR,
    #                                 momentum=config.TRAIN.MOMENTUM,
    #                                 weight_decay=config.TRAIN.WD,
    #                                 nesterov=config.TRAIN.NESTEROV,
    #                                 )
    # else:
    #     raise ValueError('Only Support SGD optimizer')

    # checkpoint
    best_mIoU = 0
    best_OA = 0
    best_fw_iou = 0
    last_epoch = 0

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True

    # prepare train datasets
    # data_loader
    train_dataset = FUSARMapV2_MultiScale(
        image_path=config.DATASET.TRAIN_IMAGE_ROOT,
        label_path=config.DATASET.TRAIN_LABEL_ROOT,
        num_classes=config.DATASET.NUM_CLASSES,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # torch data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)  # drop_last=True,

    # prepare test/validation data
    test_dataset = FUSARMapV2_MultiScale(
        image_path=config.DATASET.TEST_IMAGE_ROOT,
        label_path=config.DATASET.TEST_LABEL_ROOT,
        num_classes=config.DATASET.NUM_CLASSES,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL)

    # if args.distributed:
    #     test_sampler = DistributedSampler(test_dataset)
    # else:
    #     test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # training parameters setting
    epoch_iters = np.int32(train_dataset.__len__() / args.batch_size / 4)

    # train start
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH

    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # training process
        train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters, config.TRAIN.LR,
              num_iters, train_loader, criterion, optimizer, model, writer_dict, args)

        # valid process
        valid_loss, mean_IoU, mean_OA, IoU_array, fw_iou, rank = validate(config, test_loader, model, writer_dict, args, criterion)

        # save process
        if rank == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'best_OA': best_OA,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            if mean_OA > best_OA:  # save every 4 epoch
                sv_name = 'HRNetW18_' + 'epoch_' + str(epoch + 1) + '_fwiou_' + str(fw_iou)[:5] + '_OA_' + str(mean_OA)[:5] + '.pth'
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, sv_name))

            # save best_mIoU
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best_IoU.pth'))

            if mean_OA > best_OA:
                best_OA = mean_OA
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best_OA.pth'))

            if fw_iou > best_fw_iou and fw_iou > 0.4:
                best_fw_iou = fw_iou
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best_fw_iou.pth'))

            msg = 'Loss: {:.3f}, OA: {:.4f}, m_iou: {:.4f}, fw_iou: {:.4f}, Best_OA: {:.4f}, Best_fw_iou: {:.4f}'\
                .format(valid_loss, mean_OA, mean_IoU, fw_iou, best_OA, best_fw_iou)

            logging.info(msg)
            logging.info(IoU_array)

            # last epoch
            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int32((end - start) / 3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
