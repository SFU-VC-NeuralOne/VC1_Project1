import numpy as np
import torch
from AlexNetModified import LFWNet
import os                                       # utility lib. for file path, mkdir
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt

lfw_dataset_dir = 'lfw'
anno_train_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_train.txt')
anno_test_file_path = os.path.join(lfw_dataset_dir, 'LFW_annotation_test.txt')


def load_data(file_path):
    data_list = []
with open(label_file_path, "r") as f:
    for line in f:
        if line.startswith('#'):
            # ignore the line start with '#'
            continue
        tokens = line.split()          # split the line by ','
        file_name = tokens[0]     #TODO change to full file path
        land_mark_cord[0] = np.asarray(tokens[1:5])   # data_list[0]['land_mark_cord'][0] -- bounding box cord
        land_mark_cord[1] = np.asarray(tokens[5:])    # data_list[0]['land_mark_cord'][1] -- actual landmark cord
        data_list.append({'file_name': file_name, 'land_mark_cord': land_mark_cord})
    return data_list

class LFWDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        filename = item['filename']
        file_path = filename[0:re.search('\d',filename)-1]
        file_path = os.path.join(lfw_dataset_dir,file_path)
        bounding_box = item['land_mark_cord'][0]

        # TODO also add data augmentation, rescale and visualize here
        item = self.data_list[idx]
        label = np.asarray(item['label'])
        file_path = os.path.join(lfw_dataset_dir, item['file_path'])

        # Load image as gray-scale image and convert to (0, 1)
        img = np.asarray(Image.open(file_path).convert('L'), dtype=np.float32) / 255.0
        h, w = img.shape[0], img.shape[1]

        # Create image tensor
        img_tensor = torch.from_numpy(img)

        # Reshape to (1, 28, 28), the 1 is the channel size
        img_tensor = img_tensor.view((1, h, w))
        label_tensor = torch.from_numpy(label).long()  # Loss measure require long type tensor

        return img_tensor, label_tensor


def train(net, train_data_loader, validation_data_loader):
    # TODO nerual network name - net
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.params(), lr=0.01)

    train_losses = []
    valid_losses = []

    max_epochs = 6
    itr = 0

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):

            itr += 1

            # switch to train model
            net.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())  # use Variable(*) to allow gradient flow
            train_out = model.forward(train_input)  # forward once TODO what is model?

            # compute loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)

            # do the backward and compute gradients
            loss.backward()

            # update the parameters with SGD
            optimizer.step()

            # Add the tuple of （iteration, loss) into `train_losses` list
            train_losses.append((itr, loss.item()))

            if train_batch_idx % 200 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

                # Run the validation every 200 iteration:
            if train_batch_idx % 200 == 0:
                net.eval()  # [Important!] set the network in evaluation model
                valid_loss_set = []  # collect the validation losses
                valid_itr = 0

                # Do validation
                for valid_batch_idx, (valid_input, valid_label) in enumerate(validation_data_loader):
                    net.eval()
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_out = model.forward(valid_input)  # forward once

                    # compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    # TODO how many data should we put into validation?
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


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    data_list = load_data(anno_train_file_path)
    random.shuffle(data_list)
    total_items = len(data_list)

    # Training, ratio: 80%
    num_train_sets = 0.8 * total_items
    train_set_list = data_list[: int(num_train_sets)]

    # Validation, ratio: 20%
    validation_set_list = data_list[int(num_train_sets):]

    test_set_list = load_data(anno_test_file_path)

    # Create the dataloader for training and validation
    train_dataset = LFWDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=6)
    print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:',
          len(train_data_loader))

    validation_set = LFWDataset(validation_set_list)
    validation_data_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=6)
    print('Total validation set:', len(validation_set))

    # TODO optional: visualize some data

    # Train
    net = LFWNet()
    net.cuda()
    train(net, train_data_loader, validation_data_loader)
    net_state = model.state_dict()  # serialize trained model
    torch.save(net_state, os.path.join(lfw_dataset_dir, 'lfw_net.pth'))

    # Test
    test_net = LFWNet()
    test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'lfw_net.pth'))
    test_net.load_state_dict(test_net_state)
    test_net.eval()

    # TODO
    test_item = random.choice(test_set_list)
    test_img_path = os.path.join(lfw_dataset_dir, test_item['file_path'])

    img = np.asarray(Image.open(test_img_path).convert('L'), dtype=np.float32) / 255.0
    h, w = img.shape[0], img.shape[1]
    img_tensor = torch.from_numpy(img)

    # Reshape to (1, 1, 28, 28), the first 1 set the mini-batch to 1, the second is the channel size
    img_tensor = img_tensor.view((1, 1, h, w))

    # Forward for prediction
    pred = test_net.forward(img_tensor.cuda())

    # Find the label with max probability
    prob_max = torch.argmax(pred.detach(), dim=1)

    # Show the result
    plt.imshow(img, cmap='gray')
    plt.title("Label %d" % (prob_max.item()))
    plt.show()


if __name__ == '__main__':
    main()


