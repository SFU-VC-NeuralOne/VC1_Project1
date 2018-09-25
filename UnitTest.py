import unittest
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random

from PIL import ImageEnhance
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import re
import matplotlib.pyplot as plt
import FacialLandmark as fl
from AlexNetModified import lfw_net


class TestCropAugmentation(unittest.TestCase):

    def test_Crop(self):

        lfw_dataset_dir = 'lfw'
        anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
        data_list = fl.load_data(anno_train_file_path)
        item = random.choice(data_list)

        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]  # TODO normalize
        img = Image.open(file_path)
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
        img_og = np.asarray(img, dtype=np.float32)
        label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])

        h, w, c = img_og.shape[0], img_og.shape[1], img_og.shape[2]
        plt.figure(1)
        plt.imshow(img_og/255, cmap='brg')
        plt.plot(label[:, 0], label[:, 1], color='green', marker='o', linestyle='none', markersize=5,label='Label')
        print('original h w:', h, w)

        img_rescale = img_og / 255 * 2 - 1
        # plt.imshow(img, cmap='brg')
        # plt.plot(label[:, 0] * h, label[:, 1] * h, color='green', marker='o', linestyle='none', markersize=8,
        #          label='Label')

        #print('item :',item)
        #print('label:',label)

        bounding_box = fl.calculate_corp(label, h, w)
        #print('boudning box', bounding_box)
        img1 = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))
        img1 = np.asarray(img1, dtype=np.float32)
        new_h, new_w = img1.shape[0], img1.shape[1]
        label = label - np.asarray([bounding_box[0], bounding_box[1]])
        plt.figure(2)
        plt.imshow(img1 / 255, cmap='brg')
        plt.plot(label[:, 0], label[:, 1], color='green', marker='o', linestyle='none', markersize=5, label='Label')
        #label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])
        print('new h, w:', new_h, new_w)


        # fig, axs = plt.subplots(1, 3, figsize=(13, 6))
        # axs[0].imshow(img, cmap='brg')
        # axs[1].imshow(img.transpose(Image.FLIP_LEFT_RIGHT), cmap='brg')
        # axs[2].imshow(img1/255, cmap='brg')




        plt.show()

        self.assertEqual('foo'.upper(), 'FOO')

class TestFlipAugmentation(unittest.TestCase):

    def test_Flipping(self):
        lfw_dataset_dir = 'lfw'
        anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
        data_list = fl.load_data(anno_train_file_path)
        item = random.choice(data_list)

        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]  # TODO normalize
        img = Image.open(file_path)
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
        img_og = np.asarray(img, dtype=np.float32)
        h, w, c = img_og.shape[0], img_og.shape[1], img_og.shape[2]
        label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
        label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])
        img_rescale = img_og / 255 * 2 - 1

        #plt.imshow(img, cmap='brg')
        plt.imshow(img.transpose(Image.FLIP_LEFT_RIGHT), cmap='brg')
        label = fl.calculate_filp(label, h)
        print(label)
        #plt.plot(label[:, 0] * h, label[:, 1] * h, color='green', marker='o',linestyle='none', markersize=12, label='Label')
        plt.plot(label[0, 0] * h, label[0, 1] * h, color='green', marker='o', linestyle='none', markersize=12, label='Label')
        plt.show()
        self.assertEqual('foo'.upper(), 'FOO')

class TestBrightenAugmentation(unittest.TestCase):

    def test_Brightening(self):
        lfw_dataset_dir = 'lfw'
        anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
        data_list = fl.load_data(anno_train_file_path)
        item = random.choice(data_list)

        random_factor = random.uniform(0.5, 1.5)
        print('random factor:', random_factor)

        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]  # TODO normalize
        img = Image.open(file_path)
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
        img_og = np.asarray(img, dtype=np.float32)
        h, w, c = img_og.shape[0], img_og.shape[1], img_og.shape[2]
        label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
        label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])
        img_rescale = img_og / 255 * 2 - 1

        #plt.imshow(img, cmap='brg')
        brightness = ImageEnhance.Brightness(img)

        img = brightness.enhance(random_factor)
        plt.imshow(img, cmap='brg')
        plt.plot(label[:, 0] * h, label[:, 1] * h, color='green', marker='o',linestyle='none', markersize=12, label='Label')
        #plt.plot(label[0, 0] * h, label[0, 1] * h, color='green', marker='o', linestyle='none', markersize=12, label='Label')
        plt.title(random_factor)
        plt.show()
        self.assertEqual('foo'.upper(), 'FOO')

class TestRandomImage(unittest.TestCase):

    def test_RandomImage(self):
        lfw_dataset_dir = 'lfw'
        anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')
        data_list = fl.load_data(anno_train_file_path)
        item = random.choice(data_list)

        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]

        img = Image.open(file_path)
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
        img = img.resize((225, 225))  # rezie to alexnet input size
        img = np.asarray(img, dtype=np.float32)

        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img = img / 255 * 2 - 1
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.view(1, c, h, w)
        label_tensor = torch.from_numpy(label.flatten().astype(np.double))

        test_net = lfw_net()
        test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
        test_net.load_state_dict(test_net_state)

        pred = test_net.forward(img_tensor)

        pred_label = pred.cpu().detach().numpy()

        plt.title(pred )
        plt.imshow((img + 1) / 2, cmap='brg')
        plt.plot(pred_label[:, 0] * h, pred_label[:, 1] * h, color='green', marker='o', linestyle='none', markersize=12, label='Label')

        plt.show()

if __name__ == '__main__':
    unittest.main()