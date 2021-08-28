import os
import cv2
import numpy as np
import random


class FUSARMapV2_512():
    def __init__(self,
                 image_path,
                 label_path,
                 num_classes=10,
                 flip=True,
                 ignore_label=255):

        self.num_classes = num_classes
        self.flip = flip
        self.crop_size = (512, 512)
        self.image_rpath = image_path
        self.label_rpath = label_path
        self.ignore_label = ignore_label

        self.files = self.read_files()

        self.label_mapping = {0: [0, 0, 0], 1: [255, 255, 255], 255: [255, 0, 0]}

    def read_files(self):
        files = []
        name_list = os.listdir(self.image_rpath)
        name_list.sort()
        for name in name_list:
            # name = os.path.splitext(os.path.basename(label_path))[0]
            char_name = name.split('.')
            image_path = os.path.join(self.image_rpath, char_name[0] + '.tif')  # name
            label_path = os.path.join(self.label_rpath, char_name[0] + '.png')
            files.append({'img': image_path, 'label': label_path, 'name': name})

        return files

    def convert_label(self, label):
        temp = label.copy()
        label_mask = np.zeros((label.shape[0], label.shape[1]))
        for k, v in self.label_mapping.items():
            # label_mask[(((temp[:, :, 0] == k[0]) & (temp[:, :, 1] == k[1])) & (temp[:, :, 2] == k[2]))] = v
            label_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)

        return label_mask

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]  # 通道逆序
        # image = image.astype(np.float32)[:, :, np.newaxis]
        # image = np.tile(image, (1, 1, 3))
        image = image / 127.5 - 1  # [-1, 1]
        # image = image / 255.
        # image -= self.mean
        # image /= self.std
        return image

    def label_transform(self, label):
        label = label.astype(np.int32)[:, :, ::-1]  # BGR -> RGB
        return label  # int32

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

    def gen_sample(self, image, label, is_flip=True):

        image = self.input_transform(image)
        # label = self.label_transform(label)
        image, label = self.rand_crop(image, label)

        image = image.transpose((2, 0, 1))  # [0, 1, 2] -> [2, 0, 1], 下面对Width处理  # [C， H, W]

        if is_flip:
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）
            label = label[:, ::flip]

        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']
        # image = cv2.imread(item['img'], cv2.IMREAD_GRAYSCALE)  # color [H, W, 1]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)  # [H, W, BGR]
        # image = image[:, :, ::-1]
        size = image.shape  # [h, w, c]

        # label = cv2.imread(item['label'], cv2.IMREAD_GRAYSCALE)  # [H, W]
        label = cv2.imread(item['label'], cv2.IMREAD_COLOR)  # [H, W, BGR]
        label = self.label_transform(label)  # [H, W, RGB]
        label = self.convert_label(label)  # [h, w]

        image, label = self.gen_sample(image, label, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.files)
