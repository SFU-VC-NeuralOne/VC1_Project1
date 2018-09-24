import unittest
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import re
import matplotlib.pyplot as plt
import FacialLandmark as fl
class TestCropAugmentation(unittest.TestCase):

    def test_Crop(self):
        fig, axs = plt.subplots(1, 3, figsize=(13, 6))
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

        print('item :',item)
        print('label:',label)

        bounding_box = fl.calculate_corp(label, h, w)
        print('boudning box', bounding_box)
        img1 = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))
        img1 = np.asarray(img1, dtype=np.float32)
        draw = ImageDraw.Draw(img)

        axs[0].imshow(img, cmap='brg')
        axs[1].imshow(img1/255, cmap='brg')
        axs[2].imshow(img.transpose(Image.FLIP_LEFT_RIGHT), cmap='brg')



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

        plt.imshow(img, cmap='brg')
        plt.imshow(img.transpose(Image.FLIP_LEFT_RIGHT), cmap='brg')
        label = fl.calculate_filp(label, h)
        print(label)
        plt.plot(label[:, 0] * h, label[:, 1] * h, color='green', marker='o',linestyle='none', markersize=12, label='Label')
        #plt.plot(label[3, 0] * h, label[3, 1] * h, color='green', marker='o', linestyle='none', markersize=12, label='Label')
        plt.show()
        self.assertEqual('foo'.upper(), 'FOO')



if __name__ == '__main__':
    unittest.main()