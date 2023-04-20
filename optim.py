import torch
import torchfields
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from image_warping import warp_cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def assert_tensor(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).to(device)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)


def unnormalize(X, shape):
    X[0] = X[0]*shape[1]/2
    X[1] = X[1]*shape[0]/2
    return -X


def normalize(X, shape):
    X[0] = 2*X[0]/shape[1]
    X[1] = 2*X[1]/shape[0]
    return -X


def get_laplace():
    laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3)

    return laplace


class ShiftOptim(torch.nn.Module):

    def __init__(self, f, g, v0, lmbda, beta, correlation):

        super().__init__()

        self.f = assert_tensor(f)
        self.g = assert_tensor(g)

        self.ndim = v0.shape[0]
        self.shape = v0.shape[1:]

        # uncomment for random initial field
        """
        field = torch.rand(self.shape[0] * self.shape[1] * 2,  dtype=torch.float64) * 2 - 1
        field = field.reshape(2, *self.shape)
        """

        self.v0 = normalize(assert_tensor(v0).clone(), self.shape)

        hlags, wlags, _, _ = correlation.shape
        self.corrmax = correlation.view(hlags * wlags, -1).max(axis=0).values
        self.correlation = assert_tensor(correlation.reshape(*correlation.shape[:2], 1, -1).permute((3, 2, 0, 1)))

        field = self.v0.clone()
        self.weights = torch.nn.Parameter(field)

        self.lmbda = lmbda
        self.beta = beta

        self.laplace = get_laplace()

        self.to(device)

    def forward(self):
        return self.weights.field()(self.f)

    def loss(self, ghat):

        # mse loss related to the difference between the warped and the moved image
        err2 = torch.square(ghat - self.g)
        mse = torch.sqrt(err2.mean())

        # spatial derivative loss
        grad = F.conv2d(self.weights.reshape(2, 1, *self.f.shape), self.laplace, padding='same')

        # loss related to prior knowledge of the deformation computed by the localized cross correlation
        # we need to change the coordinates relative to the lags
        xy = unnormalize(self.weights.detach().clone(), self.shape)
        xy = normalize(xy, self.correlation.shape[-2:]).to(self.correlation.dtype)
        xy = -xy.reshape(2, -1, 1, 1).permute((1, 2, 3, 0))

        # interpolated correlations
        interp = F.grid_sample(self.correlation, xy, align_corners=True, mode='bilinear').flatten()

        # difference between the interpolated correlation and the maximum value, weighted by the maximum value
        diff = self.corrmax * (self.corrmax - interp) / self.corrmax.sum()

        return mse, self.lmbda * grad.norm(), self.beta * diff.norm()

    def fit(self, optimizer, n=1e3):
        losses = np.zeros((int(n), 3))
        loss = np.nan
        for i in range(int(n)):
            print(f'\riteration {i+1}/{n} = {(i+1) / n * 100:.2f}%, loss = {loss:.2f}', end='')

            ghat = self.forward()
            err, grad, diff = self.loss(ghat)
            loss = err + grad + diff
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[i] = err.detach().numpy(), grad.detach().numpy(), diff.detach().numpy()

        out = self.weights.detach().clone().numpy()

        return unnormalize(out, self.shape), losses


if __name__ == '__main__':

    shift = np.load('data/shift.npy', allow_pickle=True)
    im0 = np.load('data/spongebob_warp2_0.npy', allow_pickle=True)
    im1 = np.load('data/spongebob_warp2_5.npy', allow_pickle=True)
    vf = np.load('data/vf_warp2.npy', allow_pickle=True) * 5

    """
    shift = np.load('data/seismic_shift.npy', allow_pickle=True)
    im0 = np.load('data/seismic_data1.npy', allow_pickle=True)
    im1 = np.load('data/seismic_data2.npy', allow_pickle=True)
    vf = shift * 0
    """

    lmbda = 1e3
    beta = 0
    model = ShiftOptim(im0, im1, shift, lmbda, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out, losses = model.fit(optimizer, n=1000)

    plt.figure()
    plt.plot(losses[:, 0], ls='dashed', label='$\epsilon$')
    plt.plot(losses[:, 1], ls='dashed', label=f'$\lambda$={lmbda:.1e}')
    plt.plot(losses.sum(axis=1), c='k', label='$\Sigma$')
    plt.fill_between(np.arange(len(losses)), 0, losses.sum(axis=1), alpha=0.3, fc='tab:grey', ec='k')
    plt.xlim(0, len(losses) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')

    fig, axs = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(14, 3))
    axs[0].imshow(im0, origin='lower')
    axs[1].imshow(im1, origin='lower')
    axs[2].imshow(warp_cv2(im0, shift), origin='lower')
    axs[3].imshow(model.forward().detach(), origin='lower')
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig, axs = plt.subplots(3, 2, sharex='all', sharey='all')
    axs[0, 0].imshow(shift[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[0, 1].imshow(shift[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[1, 0].imshow(out[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[1, 1].imshow(out[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[2, 0].imshow(vf[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[2, 1].imshow(vf[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())

    plt.show()
