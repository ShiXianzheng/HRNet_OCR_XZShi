import os
import torch
import cv2
import numpy as np
import random


class FUSARMapV2_MultiScale():
    def __init__(self,
                 image_path,
                 label_path,
                 num_classes=10, multi_scale=True, flip=True,
                 ignore_label=255, base_size=1024, crop_size=(512, 512), downsample_rate=1,
                 scale_factor=16, center_crop_test=False, mean=None, std=None):

        # if not mean:
        #     self.mean = [32.656, 32.656, 32.656]
        # if not std:
        #     self.std = [41.127, 41.127, 41.127]

        if not mean:
            self.mean = [39.391, 38.184, 44.618]
        if not std:
            self.std = [49.646, 30.556, 51.206]

        self.num_classes = num_classes
        # self.class_weights = torch.FloatTensor([1., 1., 1.,
        #                                         1., 1., 1.,
        #                                         1., 1., 1.,
        #                                         1.]).cuda()
        self.image_rpath = image_path
        self.label_rpath = label_path

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.base_size = base_size
        self.scale_factor = scale_factor
        self.downsample_rate = downsample_rate
        self.center_crop_test = center_crop_test

        self.files = self.read_files()
        if self.num_classes == 9:
            self.label_mapping = {0: [0, 0, 255], 1: [0, 139, 0], 2: [0, 255, 0],
                                  3: [139, 0, 0], 4: [255, 0, 0], 5: [205, 173, 0],
                                  6: [83, 134, 139], 7: [139, 105, 20], 8: [189, 183, 107],
                                  255: [0, 0, 0]}  # 7: [0, 139, 139]
        elif self.num_classes == 10:
            self.label_mapping = {0: [0, 0, 255], 1: [0, 139, 0], 2: [0, 255, 0],
                                  3: [139, 0, 0], 4: [255, 0, 0], 5: [205, 173, 0],
                                  6: [83, 134, 139], 7: [0, 139, 139], 8: [139, 105, 20], 9: [189, 183, 107],
                                  255: [0, 0, 0]}

        # self.label_mapping = {[0, 0, 255]: 0, [0, 139, 0]: 1, [0, 255, 0]: 2,
        #                       [139, 0, 0]: 3, [255, 0, 0]: 4, [205, 173, 0]: 5,
        #                       [83, 134, 139]: 6, [0, 139, 139]: 7, [139, 105, 20]: 8,
        #                       [189, 183, 107]: 9, [178, 34, 34]: 255, [0, 0, 0]: 255}

        # self.label_mapping = {'Water': [0, 0, 255], 'Woodland': [0, 139, 0], 'Vegetation': [0, 255, 0],
        #                       'BareSoil': [139, 0, 0], 'Industry': [255, 0, 0], 'Residential': [205, 173, 0],
        #                       'Road': [83, 134, 139], 'PaddyLand': [0, 139, 139], 'PlantingLand': [139, 105, 20],
        #                       'HumanBuilt': [189, 183, 107],
        #                       'Others': [178, 34, 34], 'black': [0, 0, 0]}

    def center_crop(self, image, label):
        h, w = image.shape[:-1]
        x = int(round((w - self.crop_size[1]) / 2.))  # w
        y = int(round((h - self.crop_size[0]) / 2.))  # h
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    def read_files(self):
        files = []
        name_list = os.listdir(self.image_rpath)
        name_list.sort()
        for name in name_list:
            # name = os.path.splitext(os.path.basename(label_path))[0]
            image_path = os.path.join(self.image_rpath, name)
            label_path = os.path.join(self.label_rpath, name)
            files.append({'img': image_path, 'label': label_path, 'name': name})

        return files

    def convert_label(self, label):
        temp = label.copy()
        label_mask = np.zeros((label.shape[0], label.shape[1]))
        for k, v in self.label_mapping.items():
            # label_mask[(((temp[:, :, 0] == k[0]) & (temp[:, :, 1] == k[1])) & (temp[:, :, 2] == k[2]))] = v
            label_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)

        # putting paddyland to 255 for ignore label
        if self.num_classes == 9:
            label_mask[(((temp[:, :, 0] == 0) & (temp[:, :, 1] == 139)) & (temp[:, :, 2] == 139))] = 255

        return label_mask

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

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()  # shadow copy
        pad_h = max(size[0] - h, 0)  # 判断是否需要填充  [h, w, c]
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:  # 右下方填充
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)  # 边框

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]  # [h, w, c]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))  # pad, 填充零值
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))  # pad
        new_h, new_w = label.shape  # [h, w], mask
        x = random.randint(0, new_w - self.crop_size[1])  # w, col
        y = random.randint(0, new_h - self.crop_size[0])  # h, row
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
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
        image = image.astype(np.float32)
        image = image / 127.5 - 1  # [-1, 1]
        # image = image / 255.
        # image -= self.mean
        # image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')  # int32

    def gen_sample(self, image, label, multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0  # [0.5, 2] scale
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)
        else:
            image, label = self.rand_crop(image, label)

        if center_crop_test:
            image, label = self.image_resize(image, self.base_size, label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))  # [0, 1, 2] -> [2, 0, 1], 下面对Width处理

        if is_flip:
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）
            label = label[:, ::flip]

        if self.downsample_rate != 1:  # 下采样
            image = cv2.resize(image, None, fx=self.downsample_rate, fy=self.downsample_rate,
                               interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=self.downsample_rate, fy=self.downsample_rate,
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']
        image = cv2.imread(item['img'], cv2.IMREAD_COLOR)  # color [H, W, 3]
        # image = cv2.imread(item["img"], cv2.IMREAD_GRAYSCALE)  # [H, W]
        image = image.astype(np.float32)[:, :, ::-1]  # BGR -> RGB
        size = image.shape  # [h, w, c]

        # label = cv2.imread(item['label'], cv2.IMREAD_GRAYSCALE)  # [H, W]
        label = cv2.imread(item['label'], cv2.IMREAD_COLOR)  # [H, W, BGR]
        label = label.astype(np.int32)[:, :, ::-1]  # BGR -> RGB
        label = self.convert_label(label)  # [h, w]

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip, self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.files)

    # def inference(self, model, image, flip=False):
    #     size = image.size()
    #     pred = model(image)
    #     pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
    #     if flip:
    #         flip_img = image.numpy()[:, :, :, ::-1]
    #         flip_output = model(torch.from_numpy(flip_img.copy()))
    #         flip_output = F.upsample(input=flip_output, size=(size[-2], size[-1]), mode='bilinear')
    #         flip_pred = flip_output.cpu().numpy().copy()
    #         flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
    #         pred += flip_pred
    #         pred = pred * 0.5
    #     return pred.exp()

    # def multi_scale_inference(self, model, image, scales=None, flip=False):
    #     if scales is None:
    #         scales = [1]
    #     batch, _, ori_height, ori_width = image.size()
    #     assert batch == 1, "only supporting batchsize 1."
    #     image = image.numpy()[0].transpose((1, 2, 0)).copy()  # [h, w, c]
    #     stride_h = np.int(self.crop_size[0] * 1.0)
    #     stride_w = np.int(self.crop_size[1] * 1.0)
    #     final_pred = torch.zeros([1, self.num_classes,
    #                               ori_height, ori_width]).cuda()
    #     for scale in scales:
    #         new_img = self.multi_scale_aug(image=image,
    #                                        rand_scale=scale,
    #                                        rand_crop=False)
    #         height, width = new_img.shape[:-1]
    #
    #         if scale <= 1.0:
    #             new_img = new_img.transpose((2, 0, 1))  # [c, h, w]
    #             new_img = np.expand_dims(new_img, axis=0)
    #             new_img = torch.from_numpy(new_img)
    #             preds = self.inference(model, new_img, flip)
    #             preds = preds[:, :, 0:height, 0:width]
    #         else:
    #             new_h, new_w = new_img.shape[:-1]
    #             rows = np.int(np.ceil(1.0 * (new_h -
    #                                          self.crop_size[0]) / stride_h)) + 1
    #             cols = np.int(np.ceil(1.0 * (new_w -
    #                                          self.crop_size[1]) / stride_w)) + 1
    #             preds = torch.zeros([1, self.num_classes,
    #                                  new_h, new_w]).cuda()
    #             count = torch.zeros([1, 1, new_h, new_w]).cuda()
    #
    #             for r in range(rows):
    #                 for c in range(cols):
    #                     h0 = r * stride_h
    #                     w0 = c * stride_w
    #                     h1 = min(h0 + self.crop_size[0], new_h)
    #                     w1 = min(w0 + self.crop_size[1], new_w)
    #                     h0 = max(int(h1 - self.crop_size[0]), 0)
    #                     w0 = max(int(w1 - self.crop_size[1]), 0)
    #                     crop_img = new_img[h0:h1, w0:w1, :]
    #                     crop_img = crop_img.transpose((2, 0, 1))
    #                     crop_img = np.expand_dims(crop_img, axis=0)
    #                     crop_img = torch.from_numpy(crop_img)
    #                     pred = self.inference(model, crop_img, flip)
    #                     preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
    #                     count[:, :, h0:h1, w0:w1] += 1
    #             preds = preds / count
    #             preds = preds[:, :, :height, :width]
    #         preds = F.upsample(preds, (ori_height, ori_width), mode='bilinear')
    #         final_pred += preds
    #     return final_pred


