import torch
import torchfields
from torchvision import transforms
import torch.nn.functional as F
from UNet import UNet
import numpy as np
import matplotlib.pyplot as plt

from time import time

DEFAULT_CHANNELS = (2, 32, 64, 128, 256, 512)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_pow_two(x):
    return (x & (x - 1)) == 0


def assert_tensor(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).to(device)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)


class VoxelMorph2D(torch.nn.Module):

    def __init__(self, f, g, channels=None, lmbda=1e1):

        super().__init__()

        # ensuring that we have channels that are powers of two
        if channels is None:
            channels = DEFAULT_CHANNELS
        else:
            for i, ch in enumerate(channels):
                assert is_pow_two(ch), f'channels need to be a power of 2, error at index {i}, {ch} % 2 = {ch % 2}'

        self.channels = channels

        # resizing f and g to the same, square shape, multiple of two
        # for simplicity, we take the maximum channel number. might be a bad approach tho
        assert f.shape == g.shape, 'f and g need to be the same shape'
        self.original_shape = f.shape

        # we need f and g to be of shape (1, h, w)
        f = assert_tensor(f).reshape(1, *self.original_shape)
        g = assert_tensor(g).reshape(1, *self.original_shape)

        self.tf_in = transforms.Resize((self.channels[-1], self.channels[-1]))
        self.tf_out = transforms.Resize(self.original_shape)

        self.f = self.tf_in(f)
        self.g = self.tf_in(g)

        self.f = (self.f - self.f.min())/(self.f.max() - self.f.min())
        self.g = (self.g - self.g.min())/(self.g.max() - self.g.min())

        self.input = torch.cat((self.f, self.g), dim=0)
        self.input = self.input.reshape(1, *self.input.shape)

        # regularisation and optimisation parameters
        self.lmbda = lmbda

        # implement the UNet and check that it outputs the proper number of dimension (nd)
        self.unet = UNet(channels=self.channels, kernel_size=3, num_class=2)
        self.field = torch.zeros(self.input.shape[1:])

        self.unet.forward(self.input)

        self.laplace = self._get_laplace()

    def _get_laplace(self):
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        return laplace

    def forward(self):
        # we use the output from the unet as the deformation field
        field = self.unet(self.input)[0]
        self.field = torchfields.Field(field)

        out = self.field(self.f)
        return out

    def loss(self, ghat):

        err = torch.sum((ghat - self.g) ** 2)

        # need to try a first derivative approach too!!
        grad = self.lmbda * F.conv2d(self.field.reshape(2, 1, *self.field.shape[-2:]), self.laplace).norm()

        norm = 1e3 * self.field.norm()

        return err, grad, norm

    def fit(self, optimizer, n):
        losses = np.zeros((int(n), 3))
        loss = np.nan

        t0 = time()
        dt = 0.0
        tt = 0.0

        for i in range(int(n)):
            print(f'\riteration {i+1}/{n} = {(i+1) / n * 100:.2f}%, loss = {loss:.6f},'
                  f' step time: {dt:.2f} s, total time: {tt:.2f} s', end='')

            ghat = self.forward()
            err, grad, norm = self.loss(ghat)
            loss = err + grad + norm
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            t1 = time()
            dt = t1 - t0
            t0 = t1
            tt += dt

            losses[i] = err.detach().numpy(), grad.detach().numpy(), norm.detach().numpy()

        out = self.field.detach().clone().numpy()

        return self.tf_out(out), losses


if __name__ == '__main__':

    shift = np.load('data/shift.npy', allow_pickle=True)
    im0 = np.load('data/spongebob_warp2_0.npy', allow_pickle=True)
    im1 = np.load('data/spongebob_warp2_5.npy', allow_pickle=True)
    vf = np.load('data/vf_warp2.npy', allow_pickle=True) * 5

    lmbda = 1e4
    model = VoxelMorph2D(im0, im1, lmbda=lmbda)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    out, losses = model.fit(optimizer, n=5000)
    print()

"""
fig, axs = plt.subplots(1, 5, sharex='all', sharey='all', figsize=(18, 4))
axs[0].imshow(self.f[0], origin='lower')
axs[0].set_title('original image', loc='left')

axs[1].imshow(self.g[0], origin='lower')
axs[1].set_title('moved image', loc='left')

axs[2].imshow(ghat[0].detach(), origin='lower')
axs[2].set_title('warped image', loc='left')

axs[3].imshow(self.field[0].detach()*512, origin='lower', vmin=-1, vmax=1)
axs[3].set_title('v$_x$ ', loc='left')

axs[4].imshow(self.field[1].detach()*512, origin='lower', vmin=-1, vmax=1)
axs[4].set_title('v$_y$ ', loc='left')
"""

