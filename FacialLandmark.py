import numpy as np
import torch
from AlexNetModified import lfw_net
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from torch.autograd import Variable
import matplotlib.pyplot as plt
import re

lfw_dataset_dir = 'lfw'
anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
anno_test_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')
train_learning_rate = 0.000000005
training_mode = False
improve_model = True
alexnet_input_size = 225


def load_data(file_path):
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) == 0:
                continue
            img_file_name = tokens[0]
            img_dir_name = img_file_name[0:re.search('\d', img_file_name).start() - 1]
            img_file_path = os.path.join(lfw_dataset_dir, img_dir_name, img_file_name)
            cords = [[], []]
            cords[0] = np.asarray(tokens[1:5], dtype=np.double)      # bounding box cords
            cords[1] = np.asarray(tokens[5:], dtype=np.double)       # landmark cords
            data_list.append({'file_path': img_file_path, 'cords': cords})
        return data_list


class LFWDataset(Dataset):
    def __init__(self, data_list, augment_data=False):
        self.data_list = data_list
        self.augment_data = augment_data

    def __len__(self):
        return len(self.data_list) * 5 if self.augment_data else len(self.data_list)

    def __getitem__(self, idx):
        length = len(self.data_list)
        item = self.data_list[idx % length]
        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]

        data_augmentation_choices = [
            [True,  True,  True],
            [False, True,  True],
            [True,  False, True],
            [True,  True,  False],
            [False, False, True],
            [True,  False, False],
            ]

        random_cropping, horizontal_flipping, adjust_brightness = random.choice(data_augmentation_choices)\
            if idx >= length and self.augment_data else [False, False, False]

        img = Image.open(file_path)

        label = label.reshape(7, 2)

        if random_cropping:
            bounding_box_w = bounding_box[2] - bounding_box[0]
            bounding_box_h = bounding_box[3] - bounding_box[1]
            offsets = [random.random() / 10 * bounding_box_w, random.random() / 10 * bounding_box_h,
                       - random.random() / 10 * bounding_box_w, - random.random() / 10 * bounding_box_h]
            bounding_box = bounding_box + offsets
            min_label_cords = [np.min(label[:, 0]), np.min(label[:, 1])]
            max_label_cords = [np.max(label[:, 0]), np.max(label[:, 1])]
            bounding_box[0] = min(bounding_box[0], min_label_cords[0])
            bounding_box[1] = min(bounding_box[1], min_label_cords[1])
            bounding_box[2] = max(bounding_box[2], max_label_cords[0])
            bounding_box[3] = max(bounding_box[3], max_label_cords[1])

        # crop to bounding box
        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))

        # label consists of 7 cords; normalize the values to [0, 1]
        # [0,0]canthus_rr_x     [0,1]canthus_rr_y
        # [1,0]canthus_rl_x     [1,1]canthus_rl_y
        # [2,0]canthus_lr_x     [2,1]canthus_lr_y
        # [3,0]canthus_ll_x     [3,1]canthus_ll_y
        # [4,0]mouth_corner_r_x [4,1]mouth_corner_r_y
        # [5,0]mouth_corner_l_x [5,1]mouth_corner_l_y
        # [6,0]nose_x           [6,1]nose_y
        label = label - np.asarray([bounding_box[0], bounding_box[1]])
        label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])

        if horizontal_flipping:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label[:, 0] = 1 - label[:, 0]
            # swap the following cords:
            # canthus_rr with canthus_ll
            # canthus_rl with canthus_lr
            # mouth_corner_r with mouth_corner_l
            label[0, :], label[3, :] = label[3, :], label[0, :].copy()
            label[1, :], label[2, :] = label[2, :], label[1, :].copy()
            label[4, :], label[5, :] = label[5, :], label[4, :].copy()

        if adjust_brightness:
            orig_brightness = 1.0
            lowest_brightness = 0.4
            highest_brightness = 1.8
            min_diff = 0.2
            new_brightness = random.choice([random.randint(lowest_brightness * 10, (orig_brightness - min_diff) * 10) / 10,
                                            random.randint((orig_brightness + min_diff) * 10, highest_brightness * 10) / 10])
            img = ImageEnhance.Brightness(img).enhance(new_brightness)

        # resize and normalize pixel values to [-1, 1]
        img = img.resize((alexnet_input_size, alexnet_input_size))

        # plt.imshow(img)
        # labels = ['canthus_rr', 'canthus_rl', 'canthus_lr', 'canthus_ll', 'mouth_corner_r', 'mouth_corner_l', 'nose']
        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
        # for i in range(len(labels)):
        #     plt.plot(label[i, 0] * alexnet_input_size, label[i, 1] * alexnet_input_size, color=colors[i], marker='o',
        #              linestyle='none', markersize=12, label=labels[i])
        # plt.title('img test')
        # plt.legend()
        # plt.show()

        img = np.asarray(img, dtype=np.double)
        img = img / 255.0 * 2 - 1

        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.view(c, h, w)
        label_tensor = torch.from_numpy(label.ravel())

        # z, x, y = img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]
        # img_tensor = img_tensor.view(x, y, z)
        # img = img_tensor.cpu().numpy()
        # img = (img + 1) / 2
        # plt.imshow(img, cmap='brg')
        # plt.show()

        return img_tensor, label_tensor


def train(net, train_data_loader, validation_data_loader):
    net.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=train_learning_rate)

    train_losses = []
    valid_losses = []

    max_epochs = 5
    itr = 0

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):
            net.train()
            optimizer.zero_grad()

            train_input = Variable(train_input.cuda())
            train_out = net.forward(train_input)

            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)
            loss.backward()
            optimizer.step()
            train_losses.append((itr, loss.item()))

            if itr % 50 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

            # Run validation every 150 iterations:
            if itr % 150 == 0:
                net.eval()
                valid_loss_set = []

                for valid_batch_idx, (valid_input, valid_label) in enumerate(validation_data_loader):
                    valid_input = Variable(valid_input.cuda())
                    valid_out = net.forward(valid_input)

                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
                valid_losses.append((itr, avg_valid_loss))
            itr += 1

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)

    plt.plot(train_losses[:, 0],  # iteration
             train_losses[:, 1])  # loss value
    plt.plot(valid_losses[:, 0],  # iteration
             valid_losses[:, 1])  # loss value
    plt.show()


def run_test_set(net, test_data_loader):
    net.cuda()
    net.eval()
    criterion = torch.nn.MSELoss()
    distances = []
    itr = 0

    for idx, (test_input, label) in enumerate(test_data_loader):
        itr += 1
        test_input = Variable(test_input.cuda())
        test_output = net.forward(test_input)
        loss = criterion(test_output, Variable(label.cuda()))

        if itr % 200 == 0:
            print('Itr: %d Loss: %f' % (itr, loss.item()))

        label_np = label.numpy().reshape(7, 2)
        output_np = test_output.detach().cpu().numpy().reshape(7, 2)
        distances.append(np.linalg.norm(output_np - label_np, axis=1))

    distances = np.asarray(distances)
    plt.xlabel('Radius')
    plt.ylabel('Detected Ratio %')
    plt.title("Avg. Percentage of Detected Key-points")
    for i in range(1500):
        radius = i / 8000.0
        accuracy = (distances < radius).sum() / np.size(distances)
        plt.plot(radius, accuracy * 100, color='red', marker='o', markersize=2)
    plt.show()


def run_one_test(net, test_data_loader):
    net.cuda()
    net.eval()

    idx, (image, label) = next(enumerate(test_data_loader))
    test_output = net.forward(image.cuda())

    label = label.cpu().numpy().reshape(7, 2)
    test_output = test_output.detach().cpu().numpy().reshape(7, 2)

    n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    image = image.view(h, w, c)
    image = image.cpu().numpy()
    image = (image + 1) / 2
    plt.imshow(image, cmap='brg')

    plt.plot(label[:, 0] * alexnet_input_size, label[:, 1] * alexnet_input_size, color='green', marker='o',
             linestyle='none', markersize=12, label='Label')
    plt.plot(test_output[:, 0] * alexnet_input_size, test_output[:, 1] * alexnet_input_size, color='blue', marker='o',
             linestyle='none', markersize=12, label='Predicted')
    plt.legend()
    plt.show()


def main():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    if training_mode:
        net = lfw_net()
        if improve_model:
            net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
            net.load_state_dict(net_state)
        data_list = load_data(anno_train_file_path)
        random.shuffle(data_list)
        num_total_items = len(data_list)

        train_set_ratio = 0.8
        num_train_sets = train_set_ratio * num_total_items
        train_set_list = data_list[: int(num_train_sets)]
        validation_set_list = data_list[int(num_train_sets):]

        # Create dataloaders for training and validation
        train_dataset = LFWDataset(train_set_list, augment_data=True)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=350,
                                                        shuffle=True,
                                                        num_workers=6)
        print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:',
              len(train_data_loader))

        validation_dataset = LFWDataset(validation_set_list)
        validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                             batch_size=32,
                                                             shuffle=True,
                                                             num_workers=6)
        print('Total validation items:', len(validation_dataset))

        # train_dataset.__getitem__(22222)

        train(net, train_data_loader, validation_data_loader)
        torch.save(net.state_dict(), os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
    else:
        test_net = lfw_net(load_alex_net=False)
        test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
        test_net.load_state_dict(test_net_state)

        test_set_list = load_data(anno_test_file_path)
        test_dataset = LFWDataset(test_set_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=6)
        print('Total test items:', len(test_data_loader))

        # run_test_set(test_net, test_data_loader)
        run_one_test(test_net, test_data_loader)


if __name__ == '__main__':
    main()


