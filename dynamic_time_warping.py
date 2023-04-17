
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def time_shift(t, y, shift, axis=-1):
    from scipy.interpolate import interp1d
    shift_t = t - shift
    shift_y = interp1d(shift_t, y, bounds_error=False, fill_value='extrapolate', axis=axis)(t)

    return shift_y


def err(f, g, maxlag=40, plot=False):
    n = len(g)

    e = np.zeros((2 * maxlag + 1, n))

    for i, lag in enumerate(range(-maxlag, maxlag + 1)):
        shift = np.roll(g, lag)
        err = (shift - f)**2
        e[i] = err
        if plot:
            plt.figure()
            plt.title(f'Lag is {lag}')
            plt.plot(f)
            plt.plot(g, ls='dashed')
            plt.plot(shift)

            plt.fill_between(np.arange(len(f)), f, shift, alpha=0.3, fc='red')
            plt.plot(shift)
            plt.twinx()
            plt.plot(err, c='red')
            #plt.ylim(-0.01*err.max(), None)

    return normalize(e)


def normalize(x):
    x = (x - x.min())/(x.max() - x.min())
    return x


def accumulation(e):
    d = e.copy()
    nlag, ni = e.shape

    for i in range(1, ni):
        for l in range(nlag):
            lower_lag_index = max(l - 1, 0)
            upper_lag_index = min(l + 1, nlag - 1)

            d[l, i] += d[lower_lag_index:upper_lag_index+1, [i - 1]].min()
    return d


def backtracking(d):
    nlag, ni = d.shape
    lags = np.zeros(ni)
    l = d[:, -1].argmin()
    lags[-1] = l

    for i in range(ni - 1, -1, -1):
        lower_lag_index = max(l - 1, 0)
        upper_lag_index = min(l + 1, nlag - 1)

        l = d[lower_lag_index:upper_lag_index + 1, i - 1].argmin() + l - 1
        lags[i - 1] = l
    return lags


def accumulation2(e, b=1, dir=1):

    # fully inspired from Dave Hale's work:
    # https://github.com/dhale/jtk/blob/master/core/src/main/java/edu/mines/jtk/dsp/DynamicWarping.java

    d = e.copy()
    nlag, ni = e.shape

    i0, iend, step = (0, ni - 1, 1) if dir > 0 else (ni - 1, 0, -1)

    for i in range(i0, iend + step, step):

        i_index = max(0, min(ni - 1, i - step))
        b_index = max(0, min(ni - 1, i - step * b))

        for l in range(nlag):
            lower_lag_index = l - 1 if l != 0 else 0
            upper_lag_index = l + 1 if l != nlag - 1 else nlag - 1

            d1 = d[lower_lag_index, b_index]
            d2 = d[l, i_index]
            d3 = d[upper_lag_index, b_index]

            for k in range(b_index, i_index, -step):
                d1 += e[lower_lag_index, k]
                d3 += e[upper_lag_index, k]

            d[l, i] += min(d1, d2, d3)
    return d


def accumulation_old(e, b=1, dir=None):

    # fully inspired from Dave Hale's work:
    # https://github.com/dhale/jtk/blob/master/core/src/main/java/edu/mines/jtk/dsp/DynamicWarping.java

    d = e.copy()
    nlag, ni = e.shape

    for i in range(0, ni):
        i_index = max(0, min(ni, i-1))
        b_index = max(0, min(ni, i-b))

        for l in range(nlag):
            # this we do not change according to b
            lower_lag_index = max(l - 1, 0)
            upper_lag_index = min(l + 1, nlag - 1)

            d1 = d[lower_lag_index, b_index]
            d2 = d[l, i_index]
            d3 = d[upper_lag_index, b_index]

            for k in range(b_index, i_index, -1):
                d1 += e[lower_lag_index, k]
                d3 += e[upper_lag_index, k]

            d[l, i] += min(d1, d2, d3)
    return d


def backtracking2(d, e, b=1, dir=-1):
    # fully inspired from Dave Hale's work:
    # https://github.com/dhale/jtk/blob/master/core/src/main/java/edu/mines/jtk/dsp/DynamicWarping.java
    ob = 1.0/b

    nlag, ni = d.shape
    lags = np.zeros(ni)

    i0, iend, step = (0, ni - 1, 1) if dir > 0 else (ni - 1, 0, -1)

    l = d[:, i0].argmin()
    lags[i0] = l

    i = i0
    while i != iend:
        i_index = max(0, min(ni - 1, i + step))
        b_index = max(0, min(ni - 1, i + step * b))

        lower_lag_index = l - 1 if l != 0 else 0
        upper_lag_index = l + 1 if l != nlag - 1 else nlag - 1

        d1 = d[lower_lag_index, b_index]
        d2 = d[l, i_index]
        d3 = d[upper_lag_index, b_index]

        for k in range(i_index, b_index, step):
            d1 += e[lower_lag_index, k]
            d3 += e[upper_lag_index, k]

        l = np.argmin([d1, d2, d3]) + l - 1
        i += step
        lags[i] = l

        if lower_lag_index == l or l == upper_lag_index:
            dl = (lags[i] - lags[i - step]) * ob
            lags[i] = lags[i - step] + dl

            for k in range(i_index, b_index, -1):
                i += step
                lags[i] = lags[i - step] + dl

    return lags


def smooth_error(e, b):
    ef = accumulation2(e, b, dir=1)
    er = accumulation2(e, b, dir=-1)
    e = ef + er - e
    return e


def dtw(y1, y2, maxlag, b=1, dir=1, full=True):
    e = err(y1, y2, maxlag=maxlag)
    e = smooth_error(e, b)

    d = accumulation2(e, b, dir=dir)
    lag = backtracking2(d, e, b, dir=-dir) - maxlag
    if full:
        return lag, d, e
    else:
        return lag


def simple_diw(im1, im2, maxlag, b=1):

    w, h = im1.shape
    lags = np.zeros((2, w, h))
    for i in range(w):
        y1 = im1[i]
        y2 = im2[i]

        lag = dtw(y1, y2, maxlag, b, full=False)
        lags[0, i] = lag

    for j in range(h):
        y1 = im1[:, j]
        y2 = im2[:, j]

        lag = dtw(y1, y2, maxlag, b, full=False)
        lags[1, :, j] = lag

    return lags


def diw(im1, im2, maxlag, bx=1, by=1):

    w, h = im1.shape

    lags = np.zeros((2, w, h))

    # shifts in one direction
    errors_h = np.zeros((h, 2 * maxlag + 1, w))

    for j in range(h):
        y1 = im1[:, j]
        y2 = im2[:, j]
        e = err(y1, y2, maxlag)
        errors_h[j] = smooth_error(e, by)  # we first smooth the error in the vertical direction

    for i in range(w):
        errors_h[:, :, i] = smooth_error(errors_h[:, :, i].T, bx).T

    for j in range(h):
        d = accumulation2(errors_h[j], by, dir=1)
        l = backtracking2(d, errors_h[j], by, dir=-1)
        lags[1, :, j] = l

    # shifts in one direction
    errors_w = np.zeros((w, 2 * maxlag + 1, h))

    for i in range(w):
        y1 = im1[i]
        y2 = im2[i]

        e = err(y1, y2, maxlag)
        errors_w[i] = smooth_error(e, bx)  # we first smooth the error in the horizontal direction

    for j in range(h):  # then we smooth it in the vertical direction
        errors_w[:, :, j] = smooth_error(errors_w[:, :, j].T, by).T

    for i in range(w):  # finally we compute distance and do the backtracking
        d = accumulation2(errors_w[i], bx, dir=1)
        l = backtracking2(d, errors_w[i], bx, dir=-1)
        lags[0, i] = l

    return lags - maxlag


def diw_tests(im1, im2, maxlag, bx=1, by=1):
    w, h = im1.shape

    lags = np.zeros((2, w, h))

    # shifts in one direction
    errors_h = np.zeros((h, 2 * maxlag + 1, w))

    for j in range(h):
        y1 = im1[:, j]
        y2 = im2[:, j]
        e = err(y1, y2, maxlag)
        errors_h[j] = smooth_error(e, by)  # we first smooth the error in the vertical direction

    for i in range(w):
        errors_h[:, :, i] = smooth_error(errors_h[:, :, i].T, bx).T

    for j in range(h):
        d = accumulation2(errors_h[j], by, dir=1)
        l = backtracking2(d, errors_h[j], by, dir=-1)
        lags[1, :, j] = l

    # shifts in one direction
    errors_w = np.zeros((w, 2 * maxlag + 1, h))

    for i in range(w):
        y1 = im1[i]
        y2 = im2[i]

        e = err(y1, y2, maxlag)
        errors_w[i] = smooth_error(e, bx)  # we first smooth the error in the horizontal direction

    for j in range(h):  # then we smooth it in the vertical direction
        errors_w[:, :, j] = smooth_error(errors_w[:, :, j].T, by).T

    for i in range(w):  # finally we compute distance and do the backtracking
        d = accumulation2(errors_w[i], bx, dir=1)
        l = backtracking2(d, errors_w[i], bx, dir=-1)
        lags[0, i] = l

    from matplotlib.widgets import Slider

    fig1, axs = plt.subplots(1, 2)

    i0 = 10
    j0 = 10
    sliderw_ax = fig1.add_axes([0.05, 0.9, 0.4, 0.05], transform=axs[0].transAxes.inverted())
    sliderw1 = Slider(ax=sliderw_ax, valmin=0, valmax=h - 1, valinit=i0, valstep=1, label='w')

    sliderh_ax = fig1.add_axes([0.55, 0.9, 0.4, 0.05], transform=axs[1].transAxes.inverted())
    sliderh1 = Slider(ax=sliderh_ax, valmin=0, valmax=w - 1, valinit=j0, valstep=1, label='h')

    imw = axs[0].imshow(errors_h[i0, :, :])
    l1, = axs[0].plot(lags[1, :, i0], c='w')
    imh = axs[1].imshow(errors_h[:, :, j0])

    def updatew1(val):
        imw.set_data(errors_h[val, :, :])
        l1.set_ydata(lags[1, :, val])
        fig1.canvas.draw()

    def updateh1(val):
        imh.set_data(errors_h[:, :, val])
        fig1.canvas.draw()

    sliderh1.on_changed(updateh1)
    sliderw1.on_changed(updatew1)

    fig2, axs = plt.subplots(1, 2)

    i0 = 10
    j0 = 10
    sliderw_ax = fig2.add_axes([0.05, 0.9, 0.4, 0.05], transform=axs[0].transAxes.inverted())
    sliderw2 = Slider(ax=sliderw_ax, valmin=0, valmax=w - 1, valinit=i0, valstep=1, label='w')

    sliderh_ax = fig2.add_axes([0.55, 0.9, 0.4, 0.05], transform=axs[1].transAxes.inverted())
    sliderh2 = Slider(ax=sliderh_ax, valmin=0, valmax=h - 1, valinit=j0, valstep=1, label='h')

    imw = axs[0].imshow(errors_w[i0, :, :])
    lw2, = axs[0].plot(lags[0, i0], c='w')
    imh = axs[1].imshow(errors_w[:, :, j0])

    def updatew2(val):
        imw.set_data(errors_w[val, :, :])
        lw2.set_ydata(lags[0, val])
        fig2.canvas.draw()

    def updateh2(val):
        imh.set_data(errors_w[:, :, val])
        fig2.canvas.draw()

    sliderh2.on_changed(updateh2)
    sliderw2.on_changed(updatew2)
    plt.show()

    return lags - maxlag

if __name__ == '__main__':
    from time import time

    np.random.seed(42069)

    im1 = plt.imread('data/spongebob.png')[::-1, :, 0]
    warp3 = np.load('data/warp3.npy', allow_pickle=True)
    warp10 = np.load('data/warp10.npy', allow_pickle=True)
    vf = np.load('data/vf.npy')

    maxlag = 100
    t0 = time()
    lags = diw(im1, warp3, maxlag, bx=1, by=1)
    t1 = time()
    print(f'Took {t1 - t0:.2f} seconds')

    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')
    axs[0, 0].imshow(im1, origin='lower')
    axs[1, 0].imshow(warp3, origin='lower')
    axs[0, 1].imshow(vf[0], origin='lower')
    axs[1, 1].imshow(vf[1], origin='lower')
    axs[0, 2].imshow(lags[1], origin='lower')
    axs[1, 2].imshow(lags[0], origin='lower')
    plt.show()

    """ 
    # simple 1D example, seems to work well
       
    # generation of a dummy time series
    t = np.linspace(0, 1000, 1000)
    y = np.sin(2*np.pi*t/500)

    # we compute a given shift to apply to the time series
    shiftf = 100*np.sin(2*np.pi*np.arange(len(t))/1000)
    shifty = time_shift(t, y, shiftf)
    noise = 0.1*np.random.randn(2, len(t))

    y = y + noise[0]
    shifty = shifty + noise[1]

    maxlag = 200
    b = 1
    lags, d, e = dtw(y, shifty, maxlag=maxlag, b=b)

    shift_hat = time_shift(t, y, lags)

    # we can plot what the error matrix looks like
    fig, axs = plt.subplots(2, 1, sharex='col')
    axs[0].plot(t, y)
    axs[0].plot(t, shifty, c='tab:orange')
    axs[0].plot(t, shift_hat, c='tab:orange', ls='dashed')
    step = 25
    for i, l in enumerate(lags[::step].astype(int)):
        i = int(i * step)
        x0, y0 = t[i], y[i]
        x1, y1 = t[i - l], shifty[i - l]
        if (x1 - x0) < 0.9 * t.max():
            axs[0].plot([x0, x1], [y0, y1], c='k', alpha=0.3)

    axs[1].imshow(d, extent=[0, len(t), -maxlag, maxlag], aspect='auto', interpolation='none', origin='lower')
    axs[1].plot(shiftf, c='w', ls='dashed', lw=1)
    axs[1].plot(lags, c='w')
    fig.subplots_adjust(hspace=0)
    axs[0].set_ylabel('Original signals')
    axs[1].set_ylabel('lag $l$')
    plt.show()
    """

    """
    maxlag = 100
    e = err(y + noise[0], shifty + noise[1], maxlag=maxlag)#, plot=True)
    b = 1
    dist = accumulation(e)
    dist2 = accumulation2(e, b=b)
    lags = backtracking(dist) - maxlag
    lags2 = backtracking2(dist2, e, b=b) - maxlag
    """

    """
    
    from matplotlib.widgets import Slider

    fig, axs = plt.subplots(1, 2)

    i0 = 10
    j0 = 10
    sliderw_ax = fig.add_axes([0.05, 0.9, 0.4, 0.05], transform=axs[0].transAxes.inverted())
    sliderw = Slider(ax=sliderw_ax, valmin=0, valmax=w - 1, valinit=i0, valstep=1, label='w')

    sliderh_ax = fig.add_axes([0.55, 0.9, 0.4, 0.05], transform=axs[1].transAxes.inverted())
    sliderh = Slider(ax=sliderh_ax, valmin=0, valmax=h - 1, valinit=j0, valstep=1, label='h')

    imw = axs[0].imshow(errors[i0, :, :])
    imh = axs[1].imshow(errors[:, :, j0])

    def updatew(val):
        imw.set_data(errors[val, :, :])
        fig.canvas.draw()

    def updateh(val):
        imh.set_data(errors[:, :, val])
        fig.canvas.draw()

    sliderh.on_changed(updateh)
    sliderw.on_changed(updatew)
    plt.show()
    """