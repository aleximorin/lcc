from UNet import UNet
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils import data


class SeismicDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path

        tf = torchvision.transforms.Compose((transforms.Grayscale(),
                                             transforms.Resize(128),
                                             transforms.ToTensor()))
        self.image_folder = ImageFolder(path, transform=tf)

        self.images_index = torch.tensor(self.image_folder.targets) == 0
        self.masks_index = ~self.images_index
        self.length = self.images_index.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        X = self.image_folder[index][0]
        y = self.image_folder[index + self.length][0]

        return X, y


def mean_iou(pred, true):
    pass


if __name__ == '__main__':

    np.random.seed(42069)

    # declaration of path names
    path = 'unet/competition_data/'
    train_path = path + 'train/'
    test_path = path + 'test/'

    # creation of dataloaders with pytorch objects
    train_dataset = SeismicDataset(train_path)
    training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = SeismicDataset(test_path)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    _, h, w = test_dataset.image_folder[0][0].shape

    # instantiation of the unet model
    unet = UNet(channels=(1, 16, 32, 64, 128),
                kernel_size=3, num_class=1,
                sigmoid_max=1.0, sigmoid_min=-1.0)

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    err_func = torch.nn.BCELoss()

    max_epochs = 15
    total_loss = []
    for epoch in range(max_epochs):
        losses = []
        for i, (local_batch, local_mask) in enumerate(training_generator):

            out = unet(local_batch)
            loss = err_func(out, local_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.detach())

            print(f'\repoch {epoch}, iteration {i}, max loss: {np.max(losses):.4f}, min loss: {np.min(losses):.4f}, mean loss: {np.mean(losses):.4f}', end='')

        if epoch % 5 == 0:
            torch.save(unet.state_dict(), f'unet/weights/weights_epoch{epoch}.p')

        total_loss.append(losses)
        print()
    torch.save(unet.state_dict(), f'unet/weights/weights_final.p')
    print()

