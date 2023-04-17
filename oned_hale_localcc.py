import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, convolve
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.widgets import Slider


def _get_h(f, g, l, return_shifted=False):
    n_s = len(f)
    h = np.zeros(len(f))

    if l > 0:
        f2 = f[l:]
        g2 = g[:n_s - l]
        h[l:] = f2 * g2
    else:
        l = -l
        f2 = f[:n_s - l]
        g2 = g[l:]
        h[:n_s-l] = f2 * g2

    if return_shifted:
        if h[l] == 0:
            pad = (l, 0)
        else:
            pad = (0, l)
        f2 = np.pad(f2, pad)
        g2 = np.pad(g2, pad)
        return h, f2, g2

    return h


def cfg_1d(f, g, l, window):
    h = _get_h(f, g, l)
    conv = convolve(h, window, mode='same')
    return conv


def windowed_1d_cc(f, g, lags, sigma=3.0, return_convolutions=False):
    nw = 3 * sigma + 1 if sigma % 2 == 0 else 3 * sigma
    x = np.arange(nw) - (nw - 1) / 2
    window = np.exp(-(x/sigma)**2)
    window = window/window.sum()

    convolutions = np.zeros((len(lags), len(f)))

    cff = 1/np.sqrt(cfg_1d(f, f, 0, window))
    cgg = 1/np.sqrt(cfg_1d(g, g, 0, window))

    for i, l in enumerate(lags):
        conv = cfg_1d(f, g, l, window)
        nconv = conv * _get_h(cff, cgg, l)
        convolutions[i] = nconv

    maxlags = np.zeros(len(f))
    for i in range(len(f)):
        cc = convolutions[:, i]
        index = cc.argmax()

        up = cc[index - 1] - cc[index + 1]
        down = 2 * cc[index - 1] + 2 * cc[index + 1] - 4 * cc[index]
        offset = up / down
        maxlags[i] = index + offset

    if return_convolutions:
        return maxlags - lags.max(), convolutions

    return maxlags - lags.max()


def time_shift(t, y, shift, axis=-1):
    from scipy.interpolate import interp1d
    from sinc_interp import sinc_interp
    'tmp[h] = sinc_interp(t, f[h], t - smooth_dlag[h])'

    shift_t = t - shift
    #shift_y = interp1d(shift_t, y, bounds_error=False, fill_value='extrapolate', axis=axis)(t)
    shift_y = sinc_interp(t, y, shift_t)

    return shift_y


if __name__ == '__main__':

    # we generate a signal and shift it by a given amount
    N = 100
    t = np.linspace(0, 5 * np.pi, N)
    dt = t[1] - t[0]
    y1 = np.sin(t)
    y2 = np.sin(t + np.pi / 4)

    # we define a range of lags to be computed
    lags = np.arange(-15, 16)

    # what if we want a non-homogeneous shift?
    shift = 0.5 * np.sin(t * np.pi / (0.5 * t.max()))
    y3 = time_shift(t, y1, shift)
    shift_hat, convolutions = windowed_1d_cc(y3, y1, lags, sigma=5, return_convolutions=True)
    shift_hat = shift_hat * dt
    y3_hat = time_shift(t, y1, shift_hat)

    # showing the shift of the signal
    fig, axs = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    fig.subplots_adjust(hspace=0.05)
    l1, = axs[0].plot(t, y1, label='Original signal')
    l2, = axs[0].plot(t, y3, label='Shifted signal', c='tab:green')
    # axs[0].plot(t, y3_hat, label='LCC approx.', ls='dashed', c='tab:green')
    axs[0].set_ylabel('Signal (-)')
    l3, = axs[1].plot(t, shift, c='tab:orange', label='True shift')
    axs[1].set_ylabel('Deformation (m)')
    axs[1].set_xlabel('x (m)')
    axs[1].set_xlim(t[0], t[-1])
    axs[0].legend((l1, l3, l2), ('Original signal', 'Deformation field', 'Shifted signal'), loc='lower center',
                  bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
    fig.savefig('progress/time_shift_1d.png', dpi=800, bbox_inches='tight')

    # creation of a plot showing the intuition
    fig, axs = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    fig.subplots_adjust(hspace=0.05)
    l1, = axs[0].plot(t, y1, label='Original signal')
    l2, = axs[0].plot(t, y3, label='Shifted signal', c='tab:green')
    axs[0].set_ylabel('Signal (-)')
    im = axs[1].imshow(convolutions, origin='lower', aspect='auto', extent=[t[0], t[-1], lags[0] * dt, lags[-1] * dt],
                       vmin=-1, vmax=1)
    x0, y0, WIDTH, HEIGHT = make_axes_locatable(axs[1]).get_position()
    cb_ax = fig.add_axes([x0 + WIDTH + 0.01 * WIDTH, y0, 0.025 * WIDTH, HEIGHT])
    plt.colorbar(im, cax=cb_ax)
    cb_ax.set_ylabel('LCC', rotation=0, ha='center')
    l4, = axs[1].plot(t, shift, c='tab:orange', label='True shift')
    axs[1].set_ylabel('Deformation (m)')
    axs[1].set_xlabel('x (m)')
    axs[1].set_xlim(t[0], t[-1])
    axs[0].legend((l1, l2, l4), ('Original signal', 'Deformation field', 'Shifted signal'),
                  loc='lower center',
                  bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)
    fig.savefig('progress/LCC1d_intuition.png', dpi=800, bbox_inches='tight')

    # creation of a plot showing the results
    fig, axs = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    fig.subplots_adjust(hspace=0.05)
    l1, = axs[0].plot(t, y1, label='Original signal')
    l2, = axs[0].plot(t, y3, label='Shifted signal', c='tab:green')
    l3, = axs[0].plot(t, y3_hat, label='LCC approx.', ls='dashed', c='tab:purple')
    axs[0].set_ylabel('Signal (-)')
    im = axs[1].imshow(convolutions, origin='lower', aspect='auto', extent=[t[0], t[-1], lags[0] * dt, lags[-1] * dt],
                       vmin=-1, vmax=1)
    x0, y0, WIDTH, HEIGHT = make_axes_locatable(axs[1]).get_position()
    cb_ax = fig.add_axes([x0 + WIDTH + 0.01 * WIDTH, y0, 0.025 * WIDTH, HEIGHT])
    plt.colorbar(im, cax=cb_ax)
    cb_ax.set_ylabel('LCC', rotation=0, ha='center')
    l4, = axs[1].plot(t, shift, c='tab:orange', label='True shift')
    axs[1].plot(t, shift_hat, label='LCC approx.', ls='dashed', c='tab:purple')
    axs[1].set_ylabel('Deformation (m)')
    axs[1].set_xlabel('x (m)')
    axs[1].set_xlim(t[0], t[-1])
    axs[0].legend((l1, l2, l4, l3), ('Original signal', 'Deformation field', 'Shifted signal', 'LCC approx.'),
                  loc='lower center',
                  bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)
    fig.savefig('progress/LCC1d.png', dpi=800, bbox_inches='tight')

    # creation of a plot showing how it works
    from scipy.integrate import trapz
    l = 0
    h, shift_f, shift_g = _get_h(y1, y3, l, return_shifted=True)
    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(10, 5))
    lf, = axs[0].plot(t, shift_f)
    lg, = axs[0].plot(t, shift_g)
    lh, = axs[1].plot(t, h, c='tab:blue')
    patch = [axs[1].fill_between(t, 0, h, fc='tab:blue', alpha=0.3)]
    sum_text = axs[1].text(0.99, 0.99, f'$\Sigma h$ = {h.sum():.2f}', transform=axs[1].transAxes, ha='right', va='top')

    axs[1].axhline(0, c='tab:blue', ls='dashed', lw=1)

    lconv, = axs[1].plot(t, convolutions[int(l - lags.min())])
    axs[2].imshow(convolutions, origin='lower', aspect='auto',
                  extent=[t[0], t[-1], lags[0] * dt, lags[-1] * dt], vmin=-1, vmax=1)
    ll = axs[2].axhline(l*dt, c='tab:red', ls='dashed')

    axs[2].plot(t, shift, c='w', label='True shift', lw=1)
    axs[2].plot(t, shift_hat, label='LCC approx.', ls='dashed', c='k', lw=1)

    x0, y0, WIDTH, HEIGHT = make_axes_locatable(axs[0]).get_position()
    slider_ax = fig.add_axes([x0, y0 + HEIGHT + 0.1 * HEIGHT, WIDTH, 0.05 * HEIGHT])

    lag_slider = Slider(ax=slider_ax, label='$l$', valmin=lags.min(), valmax=lags.max(), valinit=0, valstep=1)

    def update(l):
        h, shift_f, shift_g = _get_h(y1, y3, l, return_shifted=True)
        lf.set_ydata(shift_f)
        lg.set_ydata(shift_g)
        lh.set_ydata(h)
        ll.set_ydata((l*dt, l*dt))

        lconv.set_ydata(convolutions[int(l) - lags.min()])

        sum_text.set_text(f'$\Sigma h$ = {h.sum():.2f}')

        patch[0].remove()
        patch[0] = axs[1].fill_between(t, 0, h, fc='tab:blue', alpha=0.3, ec='tab:blue')
        fig.canvas.draw_idle()

    axs[1].set_ylim(-1.05, 1.05)
    axs[-1].set_xlim(t[0], t[-1])
    lag_slider.on_changed(update)
    plt.show()


"""
l = 5
n_s = len(f)
h = np.zeros(len(f))
if l > 0:
    subf = f[l:]
    subg = g[:n_s - l]
    h[l:] = subf * subg
else:
    l = -l
    subf = f[:n_s - l] 
    subg = g[l:]
    h[:n_s - l] = subf * subg

fig, axs = plt.subplots(3, 1, sharex='all', sharey='all')
axs[0].plot(f)
axs[0].plot(g)

axs[1].plot(subf)
axs[1].plot(subg)

axs[2].plot(h)
"""