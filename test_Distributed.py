import argparse
# import logging
import os
# import timeit
import warnings

from tqdm import tqdm

from default import _C as config
from default import update_config
from utils import create_logger

# import pprint
import torch.backends.cudnn as cudnn
import seg_hrnet
import torch
# from modelsummary import get_model_summary
# import torch.nn as nn
import numpy as np
# from function import testval, test
import cv2
# import random
import math
from PIL import Image
from torch.nn import functional as F


#  python3 test_Distributed.py --cfg experimentals_yaml/test.yaml 
def make_datasets(img_path, height=512, width=512, stride=512):
    # [h, w, s] = [112, 112, 56]
    # [h, w, s] = [112, 112, 28]
    # stride is the classification window
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # [:, :, BGR]
    img = np.asarray(img)
    img = img.astype(np.float32)[:, :, ::-1]  # 通道逆序, why? # BGR -> RGB

    Height = img.shape[0]
    Width = img.shape[1]
    print(img.shape)

    # padding 0 for image to 512
    if (Height % height == 0) and (Width % width == 0):
        print('Nice image size for slice!')
    else:
        pad_h = height - (Height % height)
        pad_w = width - (Width % width)

        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        print('Padding OK!')

    print(img.shape)

    Height2 = img.shape[0]
    Width2 = img.shape[1]

    if (Height2 % height == 0) and (Width2 % width == 0):
        print('Nice padding image size for slice!')

    n_row = math.floor((Height2 - height) / stride) + 1
    n_col = math.floor((Width2 - width) / stride) + 1

    samples = np.zeros((n_row * n_col, height, width, 3))
    K = 0
    for m in range(n_row):
        row_start = m * stride
        row_end = m * stride + height
        for n in range(n_col):
            col_start = n * stride
            col_end = n * stride + width
            img_mn = img[row_start:row_end, col_start:col_end]
            samples[K, :, :, :] = img_mn
            K += 1

    return samples.copy(), n_row, n_col, Height, Width


class FUSARMapV2_512():
    def __init__(self, image_path, flip=True):
        # self.multi_scale = multi_scale
        self.flip = flip
        self.image_rpath = image_path
        self.crop_size = (512, 512)
        self.num_classes = 9
        self.base_size = 512
        # self.scale_factor = scale_factor

        self.files, self.n_row, self.n_col, self.Height, self.Width = self.read_files()

    def read_files(self):
        return make_datasets(self.image_rpath)

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()  # shadow copy
        pad_h = max(size[0] - h, 0)  # 判断是否需要填充  [h, w, c]
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:  # 右下方填充
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)  # 边框

        return pad_image

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:-1]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1., rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)

        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def input_transform(self, image):
        # image = image.astype(np.float32)[:, :, ::-1]  # 通道逆序, why?
        # image = image.astype(np.float32)[:, :, np.newaxis]
        # image = np.tile(image, (1, 1, 3))
        image = image.astype(np.float32)
        image = image / 127.5 - 1  # [-1, 1]
        # image = image / 255.
        # image -= self.mean
        # image /= self.std
        return image

    def gen_sample(self, image, is_flip=True):
        # if multi_scale:
        #     rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0  # [0.5, 2] scale
        #     image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        # if center_crop_test:
        #     image, label = self.image_resize(image, self.base_size, label)
        #     image, label = self.center_crop(image, label)

        image = self.input_transform(image)

        image = image.transpose((2, 0, 1))  # [0, 1, 2] -> [2, 0, 1], 下面对Width处理

        if is_flip:
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）

        return image

    def __getitem__(self, index):
        item = self.files[index]
        # name = item['name']
        # image = cv2.imread(item['img'], cv2.IMREAD_COLOR)  # color [H, W, C]
        # image = cv2.imread(item["img"], cv2.IMREAD_GRAYSCALE)  # [H, W]

        # size = image.shape  # [h, w, c]

        image = self.gen_sample(item, self.flip)

        return image.copy(), self.n_row, self.n_col, self.Height, self.Width

    def __len__(self):
        return self.files.shape[0]

    def inference(self, model, image):
        pred = model(image)
        return pred  # .exp()

    def inference_flip(self, model, image, gpu, flip=False):
        size = image.size()
        pred = model(image)
        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        pred = F.interpolate(input=pred, size=(size[-2], size[-1]), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        if flip:
            flip_img = image.cpu().numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()).cuda(gpu))
            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]
            # flip_output = model(flip_img.copy().cuda())
            flip_output = F.interpolate(input=flip_output, size=(size[-2], size[-1]), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda(gpu)
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def multi_scale_inference(self, model, image, gpu, scales=None, flip=False):
        if scales is None:
            scales = [1]
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."

        image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()

        stride_h = np.int32(self.crop_size[0] * 1.0)
        stride_w = np.int32(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda(gpu)

        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale, rand_crop=False)

            height, width = new_img.shape[:-1]
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img).cuda(gpu)
                preds = self.inference_flip(model, new_img, gpu, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int32(np.ceil(1.0 * (new_h -
                                               self.crop_size[0]) / stride_h)) + 1
                cols = np.int32(np.ceil(1.0 * (new_w -
                                               self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes, new_h, new_w]).cuda(gpu)
                count = torch.zeros([1, 1, new_h, new_w]).cuda(gpu)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img).cuda(gpu)
                        pred = self.inference_flip(model, crop_img, gpu, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = F.upsample(preds, (ori_height, ori_width), mode='bilinear')
            final_pred += preds
        return final_pred

    def save_pred(self, preds, Height, Width, name, sv_path, weights_name):
        preds = np.argmax(preds, axis=2)
        print(preds.shape)
        # self.label_mapping = {'Water': [0, 0, 255], 'Woodland': [0, 139, 0], 'Vegetation': [0, 255, 0],
        #                       'BareSoil': [139, 0, 0], 'Industry': [255, 0, 0], 'Residential': [205, 173, 0],
        #                       'Road': [83, 134, 139], 'PaddyLand': [0, 139, 139], 'PlantingLand': [139, 105, 20],
        #                       'HumanBuilt': [189, 183, 107],
        #                       'Others': [178, 34, 34], 'black': [0, 0, 0]}
        # self.label_mapping = {0: [0, 0, 255], 1: [0, 139, 0], 2: [0, 255, 0],
        #                       3: [139, 0, 0], 4: [255, 0, 0], 5: [205, 173, 0],
        #                       6: [83, 134, 139], 7: [0, 139, 139], 8: [139, 105, 20],
        #                       9: [189, 183, 107], 255: [0, 0, 0]}

        background_mask = preds == 0
        hb_mask = preds == 1

        rgb = background_mask * 0 + hb_mask * 255

        rgb = rgb[:Height, :Width]
        print(rgb.shape)
        save_img = Image.fromarray(np.uint8(rgb))
        # save_img = save_img.resize((Width, Height), Image.NEAREST)
        save_img.save(os.path.join(sv_path, 'pred_' + name + '_' + weights_name[:-4] + '.png'))


def test_model(test_dataset, testloader, model, name, sv_dir, gpu, weights_name, args):
    model.eval()
    # res = []
    n_pred = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            image, n_row, n_col, Height, Width = batch
            # print(image.shape)
            img_sh = image.shape
            image = image.cuda(gpu)  # non_blocking=True

            if args.batch_size == 1:
                scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                pred = test_dataset.multi_scale_inference(model, image, gpu, scales, flip=True)
            else:
                pred = test_dataset.inference_flip(model, image, gpu, flip=True)

            pred = F.upsample(pred, (img_sh[-2], img_sh[-1]), mode='bilinear')
            pred = pred.cpu().numpy()
            # print(pred.shape)  # [batch, classes, height, width]
            # pred = np.argmax(pred, axis=1)
            # print(pred)
            # pred = list(pred)  # [1, 2, ..., batch]
            pred = pred.transpose((0, 2, 3, 1))  # [batch, height, width, channels]
            # pred = argmax(pred, axis=1)  # [B, H, W]
            # print(pred.shape)

            n_pred.append(pred)  # n -> [B, H, W]
            # print(len(n_pred))
            # n_pred += pred  # extend
            # if (i + 1) % int(n_col) == 0:
            #     res.append(n_pred)
            #     n_pred = []
        # res = res.cpu().numpy()
        # n_pred = np.asarray(n_pred)
        # print(n_pred.shape)
        # res_np = np.asarray(n_pred, dtype='uint8')
        # print(res_np.shape)
        # print(int(n_row.numpy()[0]))
        # print(int(n_col.numpy()[0]))
        # print(n_pred[-1].shape)
        row = int(n_row.numpy()[0])
        col = int(n_col.numpy()[0])
        # print(row)
        # print(col)
        stride = 512

        classes = args.classes

        height = (row - 1) * stride + img_sh[-2]
        width = (col - 1) * stride + img_sh[-1]

        pred_np = np.zeros((height, width, classes))
        print(pred_np.shape)

        for i in range(row):
            row_start = i * stride
            row_end = row_start + img_sh[-2]
            for j in range(col):
                col_start = j * stride
                col_end = col_start + img_sh[-1]
                num = i * col + j  # 第几张图片
                l_0 = num // args.batch_size
                l_1 = num % args.batch_size
                lab = n_pred[l_0][l_1]
                # print(lab.shape)
                pred_np[row_start:row_end, col_start:col_end, :] = lab  # over-lap
        # res_np = np.reshape(res_np, (int(n_row.numpy()[0]), int(n_col.numpy()[0])))
        # res_np = torch.Tensor(res_np)
        # res_np = F.upsample(res_np, (int(Height), int(Width)), mode='nearest')
        # res_np = np.resize(res_np, (int(Height), int(Width)), )
        sv_path = os.path.join(sv_dir, 'test_results_HB')
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)

        test_dataset.save_pred(pred_np, int(Height.numpy()[0]), int(Width.numpy()[0]), name, sv_path,
                               weights_name=weights_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--data',
                        default='/data/shixianzheng/2021_compete/Seg_HB/PositiveAddNegative/Valid/image',
                        # Train_slice_1500_SAR
                        metavar='DIR',
                        help='path to dataset')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('-c', '--classes', default=9, type=int,
                        metavar='N',
                        help='number of classes')

    parser.add_argument('--pretrained',
                        default='/data/shixianzheng/2021_compete/Seg_HB/output_512_w18_all/FUSARMapV2_512/hrnet_w18_train_flip_512_HB',
                        type=str, metavar='PATH',
                        help='use pre-trained model path')

    parser.add_argument('--gpu', default=3, type=int,
                        help='GPU id to use.')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def create_filename(input_dir):
    img_filename = []
    names = []
    path_list = os.listdir(input_dir)
    path_list.sort()
    for filename in path_list:
        char_name = filename.split('.')[0]
        names.append(char_name)
        file_path = os.path.join(input_dir, filename)
        img_filename.append(file_path)

    return img_filename, names


def main():
    args = parse_args()

    # 指定GPU
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for test".format(args.gpu))

    logger, final_output_dir = create_logger(config, args.cfg, 'test')
    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
    # print(model)
    # dump_input = torch.rand((1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]))
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # load model weights
    weights_name = 'HRNetW18_epoch_225_fwiou_0.947_OA_0.971.pth'
    model_state_file = os.path.join(args.pretrained, weights_name)
    # model_state_file = config.TEST.MODEL_FILE
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    # print(pretrained_dict)

    model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     print('{}'.format(k))

    # for k, v in pretrained_dict['state_dict'].items():
    #     print('{}'.format(k))
    for k, v in pretrained_dict.items():
        print('{}'.format(k))

    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                    if k in model_dict.keys()}

    # pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items()
    #                    if k in model_dict.keys()}

    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}

    # for k, _ in pretrained_dict.items():
    #     logger.info('=> loading {} from pretrained model'.format(k))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda(args.gpu)

    data_dir = args.data
    f_names, names = create_filename(data_dir)
    for i in range(len(names)):  # len(names)
        f_name = f_names[i]
        print(f_name)
        # 文件加载，文件名循环，改变保存的文件名
        test_dataset = FUSARMapV2_512(f_name, flip=False)

        # batch_size
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

        name = names[i]
        # 增加batch_size不为1的测试算法，确保能还原
        sv_dir = args.pretrained

        test_model(test_dataset, testloader, model, name=name, sv_dir=sv_dir, gpu=args.gpu, weights_name=weights_name,
                   args=args)

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    #
    # test_dataset = FUSARMapV2(image_path=args.data, flip=False)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True)
    #
    # start = timeit.default_timer()
    # test(config, test_dataset, test_loader, model, sv_dir=final_output_dir)
    #
    # end = timeit.default_timer()
    # logger.info('Mins: %d' % np.int((end - start) / 60))
    # logger.info('Done')


if __name__ == '__main__':
    main()
