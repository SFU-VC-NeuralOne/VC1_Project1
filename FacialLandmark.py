import numpy as np
import torch
from PIL import ImageEnhance

from AlexNetModified import lfw_net
import os                                       # utility lib. for file path, mkdir
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import re

lfw_dataset_dir = 'lfw'
anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
anno_test_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')
train_learning_rate = 0.001
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
            img_dir_name = img_file_name[0:re.search('\d',img_file_name).start()-1]
            img_file_path = os.path.join(lfw_dataset_dir, img_dir_name, img_file_name)
            cords = [[], []]
            cords[0] = np.asarray(tokens[1:5],dtype=np.float32)      # bounding box cords
            cords[1] = np.asarray(tokens[5:],dtype=np.float32)       # landmark cords
            data_list.append({'file_path': img_file_path, 'cords': cords})
        return data_list

def calculate_corp(label, h, w):
    label = label*h
    pass_signal = False

    while (pass_signal == False):
        x_min = np.min(label[:, 0])
        new_bounding_x1 = x_min * np.random.random(1)
        y_min = np.min(label[:, 1])
        new_bounding_y1 = y_min * np.random.random(1)

        x_max = np.max(label[:, 0])
        new_bounding_x2 = (w - x_max) * np.random.random(1) + x_max
        y_max = np.max(label[:, 1])
        new_bounding_y2 = (w - y_max) * np.random.random(1) + y_max

        if ((new_bounding_x2-new_bounding_x1)>(new_bounding_y2 - new_bounding_y1)):
            new_height = new_bounding_x2-new_bounding_x1
            new_bounding_y2 = new_bounding_y1 + new_height
        else:
            new_height = new_bounding_y2 - new_bounding_y1
            new_bounding_x2 = new_bounding_x1 + new_height

        if ((new_bounding_x2 <= h) & (new_bounding_y2 <= h)):
            pass_signal = True

    new_bb = [new_bounding_x1[0], new_bounding_y1[0], new_bounding_x2[0], new_bounding_y2[0] ]
    #return new_bounding_x1, new_bounding_y1, new_bounding_x2, new_bounding_y2
    return new_bb


def calculate_filp(label, h):
    label = label * h
    label[:,0] = h - label[:,0]
    # swap the following cords:
    # canthus_rr with canthus_ll
    # canthus_rl with canthus_lr
    # mouth_corner_r with mouth_corner_l
    label[0, :], label[3, :] = label[3, :], label[0, :].copy()
    label[1, :], label[2, :] = label[2, :], label[1, :].copy()
    label[4, :], label[5, :] = label[5, :], label[4, :].copy()

    return label/h


class LFWDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list) * 4  # original + cropping + flipping + brightness change

    def __getitem__(self, idx):
        original_length = len(self.data_list)
        item = self.data_list[idx % original_length]
        file_path = item['file_path']
        bounding_box= item['cords'][0]
        label = item['cords'][1]
        img = Image.open(file_path)

        data_augmentation_choices = [
            [True, True, True],
            [False, True, True],
            [True, False, True],
            [True, True, False],
            [False, False, True],
            [True, False, False],
            [False, True, False],
            [False, False, False]
        ]
        random_cropping, horizontal_flipping, adjust_brightness = random.choice(data_augmentation_choices)

        #     img_tensor = torch.from_numpy(img)
        #     img_tensor = img_tensor.view((1, c, h, w))
        #     label_tensor = torch.from_numpy(label).long()  # TODO

        img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))  # crop to bonding box
        img_array = np.asarray(img, dtype=np.float32)
        h, w, c = img_array.shape[0], img_array.shape[1], img_array.shape[2]
        label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
        label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])

        if random_cropping:
            bounding_box = calculate_corp(label, h, w)
            img = img.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))
            label = label.reshape(7, 2) - np.asarray([bounding_box[0], bounding_box[1]])
            label = label / np.asarray([(bounding_box[2] - bounding_box[0]), (bounding_box[3] - bounding_box[1])])

        if horizontal_flipping:     # flipping
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = calculate_filp(label, h)

        if adjust_brightness:                                   # brightness change
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(random.uniform(0.5, 1.5)) #brighten the image between 0.5 to 1.5

        img = img.resize((alexnet_input_size, alexnet_input_size))  # rezie to alexnet input size
        img= np.asarray(img, dtype=np.double)

        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img = img / 255 * 2 - 1

        img_tensor = torch.from_numpy(img)
        # if(img == None):
        #     print('file path:',file_path)
        #     print('h,w,c:', h,w,c)
        #     print('choice: ',random_cropping, horizontal_flipping, adjust_brightness)
        #     plt.imshow(img, cmap='brg')
        #     plt.show()
        img_tensor = img_tensor.view(c, h, w)
        label_tensor = torch.from_numpy(label.flatten().astype(np.double))

        return img_tensor, label_tensor


def train(net, train_data_loader, validation_data_loader):
    net.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=train_learning_rate)

    train_losses = []
    valid_losses = []

    max_epochs = 1
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

            #if train_batch_idx % 50 == 0:
            print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

            # Run the validation every 200 iteration:
            if train_batch_idx % 50 == 0:
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


                    # valid_itr += 1
                    # if valid_itr > 5:
                    #     break

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


def test(net, test_set_list):
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
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
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

    # TODO optional: visualize some data

    # Train
    net = lfw_net()
    train(net, train_data_loader, validation_data_loader)
    net_state = net.state_dict()  # serialize trained model
    torch.save(net_state, os.path.join(lfw_dataset_dir, 'lfw_net.pth'))

    # Test
    test_net = lfw_net()
    test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
    test_net.load_state_dict(test_net_state)

    # TODO allow keyboard input to trigger test()
    #test(test_net, test_set_list)


if __name__ == '__main__':
    main()


