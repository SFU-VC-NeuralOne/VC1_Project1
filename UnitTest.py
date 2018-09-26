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
        #label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])
        img_rescale = img_og / 255 * 2 - 1

        #plt.imshow(img, cmap='brg')
        plt.imshow(img.transpose(Image.FLIP_LEFT_RIGHT), cmap='brg')
        label = fl.calculate_filp(label, h)
        print(label)
        plt.plot(label[:, 0], label[:, 1], color='green', marker='o',linestyle='none', markersize=5, label='Label')
        #plt.plot(label[0, 0] * h, label[0, 1] * h, color='green', marker='o', linestyle='none', markersize=12, label='Label')
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
        test_net = lfw_net()
        test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
        test_net.load_state_dict(test_net_state)
        anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')
        data_list = fl.load_data(anno_train_file_path)
        l2_distance_list = []
        item = random.choice(data_list)
        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]
        label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
        label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])

        img = Image.open(file_path)
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
        img = img.resize((225, 225))  # rezie to alexnet input size
        img = np.asarray(img, dtype=np.float32)

        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img = img / 255 * 2 - 1
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.view(1, c, h, w)

        prediction = test_net.forward(img_tensor)
        pred_label = prediction.cpu().detach().numpy().reshape(7, 2)
        l2_distance = np.linalg.norm((pred_label - label), axis=1)
        l2_distance = np.average(l2_distance)
        l2_distance_list.append(l2_distance)

        print('predicted: ', pred_label)
        print('ground truth: ', label)
        print('l2 distance', l2_distance_list)
        l2_distance_list.sort()
        print('l2 ordered', l2_distance_list)
        plt.imshow((img + 1) / 2, cmap='brg')
        plt.plot(pred_label[:, 0] * h, pred_label[:, 1] * h, color='green', marker='o', linestyle='none', markersize=5, label='Prediction')
        plt.plot(label[:, 0]*h, label[:, 1]*h, color='blue', marker='o', linestyle='none', markersize=5, label='Ground Truth')

        plt.show()

class TestAccuracy(unittest.TestCase):

    def test_Accuracy(self):
        lfw_dataset_dir = 'lfw'
        test_net = lfw_net()
        test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
        test_net.load_state_dict(test_net_state)
        anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')
        data_list = fl.load_data(anno_train_file_path)
        l2_distance_list = []
        accuracy_plot = []
        for item in data_list[0:200]:
            file_path = item['file_path']
            bounding_box = item['cords'][0]
            label = item['cords'][1]
            label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
            label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])

            img = Image.open(file_path)
            img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
            img = img.resize((225, 225))  # rezie to alexnet input size
            img = np.asarray(img, dtype=np.float32)

            h, w, c = img.shape[0], img.shape[1], img.shape[2]
            img = img / 255 * 2 - 1
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.view(1, c, h, w)

            prediction = test_net.forward(img_tensor)
            pred_label = prediction.cpu().detach().numpy().reshape(7, 2)
            l2_distance = np.linalg.norm((pred_label - label), axis=1)
            l2_distance = np.average(l2_distance)
            l2_distance_list.append(l2_distance)

        print('l2 distance', l2_distance_list)
        l2_distance_list.sort()
        print('l2 ordered', l2_distance_list)
        for idx in range (0, len(l2_distance_list)):
            accuracy_plot.append([l2_distance_list[idx], idx/len(l2_distance_list)])
        print(accuracy_plot)
        accuracy_plot = np.asarray(accuracy_plot)
        print(accuracy_plot)
        plt.plot(np.asarray(accuracy_plot[:,0]),
                 np.asarray(accuracy_plot[:,1]))
        plt.show()

class TestNN(unittest.TestCase):

    def test_NN(self):
        lfw_dataset_dir = 'lfw'
        test_net = lfw_net()
        count = 0
        for params in test_net.parameters():
            count += 1
            if count<10:
                params.requires_grad = False
            print(params.cpu().detach().numpy().shape)

        print(count)
        print(1e-6 - 0.000001)
if __name__ == '__main__':
    unittest.main()