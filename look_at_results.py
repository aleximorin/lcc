import numpy as np
import pickle
import torch
import io
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


if __name__ == '__main__':

    path = 'out.p'

    with open(path, 'rb') as file:
        out_dict = CPU_Unpickler(file).load()

    plt.close('all')

    lcc = out_dict['lcc']
    out = out_dict['out']

    _, nt, nx = out.shape
    dt = 3e-10 * 1e9
    dx = 0.20
    t = np.arange(nt) * dt * 5
    x = np.arange(nx) * dx * 3

    vx = out[0] * dx * 3
    vy = out[1] * dt * 5

    from matplotlib import patheffects as pe

    text_params = dict(path_effects=[pe.Stroke(foreground='k', linewidth=2), pe.Normal()],
                       ha='right', va='bottom', c='w')
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(8, 8))
    vmax = np.quantile(lcc.f, 0.9)
    axs[0, 0].pcolormesh(x, t, lcc.f, vmin=-vmax, vmax=vmax, cmap='gray')
    axs[0, 1].pcolormesh(x, t, lcc.g, vmin=-vmax, vmax=vmax, cmap='gray')

    xmax = np.abs(vx).max()
    ymax = np.abs(vy).max()
    axs[1, 0].pcolormesh(x, t, vx, vmin=-xmax, vmax=xmax, cmap='coolwarm')
    axs[1, 1].pcolormesh(x, t, vy, vmin=-ymax, vmax=ymax, cmap='coolwarm')

    axu = axs[1, 0].twiny()
    axu.plot(vx.mean(axis=1), t, c='k', lw=1)

    axu.tick_params('x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True)
    axu.set_xlabel('depth-averaged displacement (m)')
    axu.xaxis.set_label_position('bottom')

    axv = axs[1, 1].twiny()
    axv.plot(vy.mean(axis=1), t, c='k', lw=1)
    axv.tick_params('x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True)
    axv.set_xlabel('depth-averaged displacement (ns)')
    axv.xaxis.set_label_position('bottom')
    axs[0, 0].invert_yaxis()

    for ax in axs[:, 0]:
        ax.set_ylabel('t (ns)')
    for ax in axs[0, :]:
        ax.set_xlabel('x (m)')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labeltop=True)
    for ax in axs[-1, :]:
        ax.tick_params(axis='x', labelbottom=False, length=0)

    axs[0, 0].text(0.99, 0.005, 'original image', transform=axs[0, 0].transAxes, **text_params)
    axs[0, 1].text(0.99, 0.005, 'original + 12 months * 20 m a$^{-1}$', transform=axs[0, 1].transAxes, **text_params)
    axs[1, 0].text(0.99, 0.005, '$u$', transform=axs[1, 0].transAxes, **text_params)
    axs[1, 1].text(0.99, 0.005, '$v$', transform=axs[1, 1].transAxes, **text_params)

    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    fig.savefig('estimated_displacements_fwd_simul.png', dpi=800, bbox_inches='tight')