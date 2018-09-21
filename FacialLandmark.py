import numpy as np
import torch
from AlexNetModified import lfw_net
import os                                       # utility lib. for file path, mkdir
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from torch.autograd import Variable
import matplotlib.pyplot as plt
import re

lfw_dataset_dir = 'lfw'
anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
anno_test_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')
train_learning_rate = 0.01


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
            img_dir_name = img_file_name[0:re.search('\d',img_file_name).start() - 1]
            img_file_path = os.path.join(lfw_dataset_dir, img_dir_name, img_file_name)
            cords = [[], []]
            cords[0] = np.asarray(tokens[1:5], dtype=np.float32)      # bounding box cords
            cords[1] = np.asarray(tokens[5:], dtype=np.float32)       # landmark cords
            data_list.append({'file_path': img_file_path, 'cords': cords})
        return data_list


class LFWDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list) * 4  # original + random cropping + flipping + brightness change

    def __getitem__(self, idx):
        length = len(self.data_list)
        item = self.data_list[idx % length]
        file_path = item['file_path']
        bounding_box = item['cords'][0]
        label = item['cords'][1]

        random_cropping = length <= idx < length * 2
        horizontal_flipping = length * 2 <= idx < length * 3
        adjust_brightness = length * 3 <= idx < length * 4

        img = Image.open(file_path)

        if random_cropping:
            bounding_box_w = bounding_box[2] - bounding_box[0]
            bounding_box_h = bounding_box[3] - bounding_box[1]
            offsets = [random.random() / 10 * bounding_box_w, random.random() / 10 * bounding_box_h,
                       - random.random() / 10 * bounding_box_w, - random.random() / 10 * bounding_box_h]
            bounding_box = bounding_box + offsets

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
        label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
        label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])

        if horizontal_flipping:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = 1 - label
            print(label)
            # swap the following cords:
            # canthus_rr with canthus_ll
            # canthus_rl with canthus_lr
            # mouth_corner_r with mouth_corner_l
            label[0, :], label[3, :] = label[3, :], label[0, :].copy()
            label[1, :], label[2, :] = label[2, :], label[1, :].copy()
            label[4, :], label[5, :] = label[5, :], label[4, :].copy()

        if adjust_brightness:
            # original brightness level is 1.0
            lowest_brightness = 0.2
            highest_brightness = 2.0
            img = ImageEnhance.Brightness(img).enhance(random.randint(lowest_brightness * 10,
                                                                      highest_brightness * 10) / 10.0)

        # resize and normalize pixel values to [-1, 1]
        img = img.resize((225, 225))
        img = np.asarray(img, dtype=np.float32)
        img = img / 255 * 2 - 1

        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.view(c, h, w)
        label_tensor = torch.from_numpy(label.ravel())
        return img_tensor, label_tensor


def train(net, train_data_loader, validation_data_loader):
    net.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.params(), lr=train_learning_rate)

    train_losses = []
    valid_losses = []

    max_epochs = 6
    itr = 0

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):
            itr += 1
            net.train()
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())
            train_out = net.forward(train_input)

            # compute loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)
            loss.backward()
            optimizer.step()
            train_losses.append((itr, loss.item()))

            if train_batch_idx % 200 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

            # Run the validation every 200 iteration:
            if train_batch_idx % 200 == 0:
                net.eval()
                valid_loss_set = []
                valid_itr = 0

                for valid_batch_idx, (valid_input, valid_label) in enumerate(validation_data_loader):
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_out = net.forward(valid_input)  # forward once

                    # compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    # TODO how many data should we put into validation? also current implementation always only validates first 5 elements
                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
                valid_losses.append((itr, avg_valid_loss))

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)

    plt.plot(train_losses[:, 0],  # iteration
             train_losses[:, 1])  # loss value
    plt.plot(valid_losses[:, 0],  # iteration
             valid_losses[:, 1])  # loss value
    plt.show()


def run_test(net, test_set_list):
    net.cuda()
    net.eval()
    test_item = random.choice(test_set_list)
    test_img_path = os.path.join(lfw_dataset_dir, test_item['file_path'])
    img = np.asarray(Image.open(test_img_path), dtype=np.float32) / 255.0   # TODO rescale to (-1, 1)
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.view((1, c, h, w))

    prediction = net.forward(img_tensor.cuda())
    prob_max = torch.argmax(prediction.detach(), dim=1)

    # Show the result TODO plot the landmarks
    plt.imshow(img)
    plt.title("Label %d" % (prob_max.item()))
    plt.show()


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    data_list = load_data(anno_train_file_path)
    random.shuffle(data_list)
    num_total_items = len(data_list)

    # Training set, ratio: 80%
    num_train_sets = 0.8 * num_total_items
    train_set_list = data_list[: int(num_train_sets)]
    validation_set_list = data_list[int(num_train_sets):]
    test_set_list = load_data(anno_test_file_path)

    # Create dataloaders for training and validation
    train_dataset = LFWDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
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

    # Train
    net = lfw_net()
    train(net, train_data_loader, validation_data_loader)
    net_state = net.state_dict()  # serialize trained model
    torch.save(net_state, os.path.join(lfw_dataset_dir, 'lfw_net.pth'))

    # Test
    test_net = lfw_net()
    test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
    test_net.load_state_dict(test_net_state)

    run_test(test_net, test_set_list)


if __name__ == '__main__':
    main()


