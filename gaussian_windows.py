import numpy as np
import torch


def gaussian_window1d(sigma):
    nw = 3 * sigma + 1 if sigma % 2 == 0 else 3 * sigma
    x = np.arange(nw) - (nw - 1) / 2
    window = np.exp(-(x/sigma)**2)
    return window/window.sum()


def gaussian_window2d(sigma):
    window = gaussian_window1d(sigma)
    window = np.outer(window, window)
    return window/window.sum()


def torch_gaussian_window1d(sigma):
    nw = 3 * sigma + 1 if sigma % 2 == 0 else 3 * sigma
    x = torch.arange(nw) - (nw - 1) / 2
    window = torch.exp(-(x/sigma)**2)

    return window/window.sum()


def torch_gaussian_window2d(sigma):
    window = torch_gaussian_window1d(sigma)
    window = torch.outer(window, window)
    window = window.reshape(1, 1, *window.shape)
    return window/window.sum()