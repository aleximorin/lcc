import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from sinc_interp import sinc_interp
import poly_interp as poly
from scipy.ndimage import convolve1d

import gaussian_windows as gw

from numba import njit
import itertools

from scipy.optimize import minimize
from image_warping import warp_cv2


def argmax2d(im):
    h, w = im.shape  # number of rows, number of columns
    index = im.argmax()  # flattened index

    j = index % w  # row
    i = index // w  # column

    return i, j


def _get_h(f, g, l1, l2):

    height, width = f.shape
    h = np.zeros((height, width))

    if l1 > 0:
        if l2 > 0:
            h[l1:, l2:] = f[l1:, l2:] * g[:height - l1, :width - l2]
        else:
            l2 = -l2
            h[l1:, :width - l2] = f[l1:, :width - l2] * g[:height - l1, l2:]
    else:
        l1 = -l1
        if l2 > 0:
            h[:height - l1, l2:] = f[:height - l1, l2:] * g[l1:, :width - l2]
        else:
            l2 = -l2
            h[:height - l1, :width - l2] = f[:height - l1, :width - l2] * g[l1:, l2:]

    return h


def _cfg(f, g, l1, l2, window):

    hfg = _get_h(f, g, l1, l2)
    out = convolve(hfg, window, mode='same')
    return out


def lcc(f, g, cff, cgg, l1, l2, window):
    conv = _cfg(f, g, l1, l2, window)
    nconv = conv * _get_h(cff, cgg, l1, l2)
    return nconv


def find_peak(im, degree, threshold):
    dw, dh = argmax2d(im)
    dh, dw, m = poly.poly_interp_newton(im,
                                        dw, dh,
                                        degree=degree,
                                        threshold=threshold)
    return dw, dh, m


class LCC2D:

    def __init__(self, f, g,
                 hlags,
                 wlags,
                 search_sigma,
                 smooth_sigma,
                 poly_degree=2,
                 threshold=1):

        assert f.shape == g.shape, 'f and g need to have the same shape'

        self.f = f
        self.g = g
        self.height, self.width = self.f.shape

        self.poly_degree = poly_degree
        self.threshold = threshold

        self.hlags = hlags
        self.wlags = wlags

        self.search_sigma = search_sigma
        self.smooth_sigma = smooth_sigma

        self.search_window = gw.gaussian_window2d(self.search_sigma)
        self.smooth_window = gw.gaussian_window1d(self.smooth_sigma)

        self.convolutions, self.shift, self.coeffs = self._windowed_2d_cc()

    def _local_convolution(self, f, g, cff, cgg):

        out_shape = len(self.hlags), len(self.wlags), self.height, self.width

        # we loop over every lag in both dimensions
        # # could be parallelized
        generator = ((f, g, cff, cgg, l1, l2, self.search_window) for (ij, (l1, l2)) in enumerate(itertools.product(self.hlags, self.wlags)))

        """from multiprocessing import Pool
        pool = Pool()

        results = pool.starmap(lcc, generator)
        convolutions = np.reshape(results, out_shape)
        """

        convolutions = np.zeros((len(self.hlags), len(self.wlags), self.height, self.width))

        for i, l1 in enumerate(self.hlags):
            for j, l2 in enumerate(self.wlags):
                convolutions[i, j] = lcc(f, g, cff, cgg, l1, l2, self.search_window)

        return convolutions

    def _polynomial_approximation(self, convolutions):

        """
        shift_shape = (2, self.height, self.width)
        coeff_shape = (self.height, self.width)

        generator = ((convolutions[:, :, h, w], self.poly_degree, self.threshold) for (h, w) in itertools.product(np.arange(self.height), np.arange(self.width)))
        from multiprocessing import Pool
        pool = Pool()
        results = pool.starmap(find_peak, generator)
        results = np.array(results)
        shift = results[:, :2].transpose().reshape(shift_shape).astype(np.float64)
        shift[0] -= self.wlags.max()
        shift[1] -= self.hlags.max()
        coeffs = results[:, -1].reshape(coeff_shape)
        
        """

        # we loop in 2d to compute subpixel displacements
        # # also could be parallelized
        shift = np.zeros((2, self.height, self.width))
        coeffs = np.zeros((self.height, self.width), dtype='object')
        for h in range(self.height):
            for w in range(self.width):
                
                dw, dh = argmax2d(convolutions[:, :, h, w])
                dh, dw, m = poly.poly_interp_newton(convolutions[:, :, h, w],
                                                    dw, dh,
                                                    degree=self.poly_degree,
                                                    threshold=self.threshold)

                shift[0, h, w] = dw - self.wlags.max()
                shift[1, h, w] = dh - self.hlags.max()
                coeffs[h, w] = m

        return shift, coeffs

    def _weighted_approximation(self, convolutions, quantile=0.99):
        shift = np.zeros((2, self.height, self.width))
        coeffs = np.zeros((self.height, self.width), dtype='object')

        xx, yy = np.meshgrid(self.wlags, self.hlags)
        X = np.array((xx.flatten(), yy.flatten()))

        for h in range(self.height):
            for w in range(self.width):
                weights = convolutions[:, :, h, w].flatten()
                weights = weights ** 2
                weights[weights < np.quantile(weights, quantile)] = 0
                weights = weights / weights.sum()
                shift[:, h, w] = np.sum(weights * X, axis=1)
                coeffs[h, w] = np.zeros(9)

        return shift, coeffs

    def _windowed_2d_cc(self):
        # this function could be split up in multiple functions to facilitate testing of different methods

        """
        # apply PEFs to the images according to what Dave says
        # # it is probably wrong however, recheck i,j,k1,k2 indices from Hale's paper?
        # f = prediction_error_filter(f, window)
        # g = prediction_error_filter(g, window)
        """

        # smoothing the images with a sigma = 1 gaussian window
        smooth_window = gw.gaussian_window2d(1)
        smooth_window = smooth_window / smooth_window.sum()
        f = self.f  # convolve(self.f, smooth_window, mode='same')
        g = self.g  # convolve(self.g, smooth_window, mode='same')

        # we can compute the normalization factors only one time
        cff = 1/np.sqrt(_cfg(f, f, 0, 0, self.search_window))
        cgg = 1/np.sqrt(_cfg(g, g, 0, 0, self.search_window))

        # convolution matrix which will store the cross-correlations
        convolutions = self._local_convolution(f, g, cff, cgg)

        # WARNING: ATTEMPT
        # here we smooth the correlations in the xy space for every lag
        #convolutions = convolve1d(convolutions, self.smooth_window, axis=-1)
        #convolutions = convolve1d(convolutions, self.smooth_window, axis=-2)

        # we loop in 2d to compute subpixel displacements
        # # also could be parallelized
        #shift, coeffs = self._weighted_approximation(convolutions)
        shift, coeffs = self._polynomial_approximation(convolutions)

        return convolutions, shift, coeffs

    def debug_plot(self, aspect=1):

        """
        # moving lags
        fig1 = plt.figure()
        ax1 = fig1.add_axes((0.1, 0.1, 0.8, 0.8))

        im = ax1.imshow(self.convolutions[int(self.hlags.max()), int(self.wlags.max())],
                       vmin=-1, vmax=1, origin='lower')

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        x0, y0, HEIGHT, WIDTH = make_axes_locatable(ax1).get_position()
        x1 = x0 + WIDTH
        y1 = y0 + HEIGHT

        l1_ax = fig1.add_axes((x0, y1 + 0.01 * HEIGHT, WIDTH, 0.05 * HEIGHT))
        l1_slider = Slider(l1_ax, '', valmin=self.wlags.min(), valmax=self.wlags.max(), valinit=0, valstep=self.wlags,
                           valfmt='$l_2$: %.f')

        l2_ax = fig1.add_axes((x1 + 0.01 * WIDTH, y0, 0.05 * WIDTH, HEIGHT))
        l2_slider = Slider(l2_ax, '', valmin=self.hlags.min(), valmax=self.hlags.max(), valinit=0, valstep=self.hlags,
                           valfmt='$l_1$: %.f',
                           orientation='vertical')

        def update_l2(l2):
            l1 = l1_slider.val
            im.set_data(self.convolutions[int(l1 - self.hlags.max()), int(l2 - self.wlags.max())])
            fig1.canvas.draw_idle()

        def update_l1(l1):
            l2 = l2_slider.val
            im.set_data(self.convolutions[int(l1 - self.hlags.max()), int(l2 - self.wlags.max())])
            fig1.canvas.draw_idle()

        l2_slider.on_changed(update_l2)
        l1_slider.on_changed(update_l1)
        """

        ##################
        # CLICKABLE FIGURE
        i0, j0 = self.height // 2, self.width // 2

        # creation of the figure and adding the main axes on which stuff will be plotted
        fig4 = plt.figure(figsize=(12, 6))
        gs = fig4.add_gridspec(2, 4)
        f_ax = fig4.add_subplot(gs[0, 0])
        g_ax = fig4.add_subplot(gs[1, 0])
        u_ax = fig4.add_subplot(gs[0, 1])
        v_ax = fig4.add_subplot(gs[1, 1])

        # main_ax is the one showing the correlation value with respect to vertical and horizontal lags
        main_ax = fig4.add_subplot(gs[:, 2:])
        main_ax.yaxis.tick_right()
        main_ax.yaxis.set_label_position('right')
        main_ax.set_xlabel('$l_2$')
        main_ax.set_ylabel('$l_1$', rotation=0)

        # f_ax and g_ax are the before and after image
        f_ax.imshow(self.f, aspect=aspect, origin='lower')
        f_ax.text(0.99, 0.99, '$f$', ha='right', va='top', transform=f_ax.transAxes)

        g_ax.imshow(self.g, aspect=aspect, origin='lower')
        g_ax.text(0.99, 0.99, '$g$', ha='right', va='top', transform=g_ax.transAxes)

        # u_ax and v_ax are the axes on which we show the estimated shifts
        u_ax.imshow(self.shift[0], aspect=aspect, origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
        u_ax.text(0.99, 0.99, '$\hat{u}$', ha='right', va='top', transform=u_ax.transAxes)

        v_ax.imshow(self.shift[1], aspect=aspect, origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
        v_ax.text(0.99, 0.99, '$\hat{v}$', ha='right', va='top', transform=v_ax.transAxes)

        # conv_im is the 2d image that will be updated for every row and column inspected
        conv_im = main_ax.imshow(self.convolutions[:, :, i0, j0], aspect='auto', origin='lower',
                                 extent=[self.wlags.min() - 0.5,
                                         self.wlags.max() + 0.5,
                                         self.hlags.min() - 0.5,
                                         self.hlags.max() + 0.5],
                                 vmin=-1, vmax=1)

        # here we want to be able to have an updating polynomial surface
        I, J = argmax2d(self.convolutions[:, :, i0, j0])
        xx, yy, z = poly.poly_surface(self.coeffs[i0, j0], I, J, degree=self.poly_degree, threshold=self.threshold)
        conv_cntrs = [main_ax.contour(xx - self.hlags.max(), yy - self.wlags.max(), z, colors='k', linewidths=0.5)]

        # those lines on main_ax show the estimated maximum value according to the polynomial surface
        lx = main_ax.axvline(self.shift[0, i0, j0], c='red', lw=0.5)
        ly = main_ax.axhline(self.shift[1, i0, j0], c='red', lw=0.5)

        # we are adding moving circles that show the moving windows
        patches = []
        from matplotlib.patches import Circle
        for ax1 in (f_ax, g_ax):
            for i in [1, 2, 3]:
                circle = Circle((j0, i0), radius=i * self.search_sigma, alpha=0.1 * (4 - i),
                                facecolor='tab:red', ec='tab:red',
                                linewidth=1.0)
                ax1.add_patch(circle)
                patches.append(circle)

        # here we are adding horizontal and vertical lines that show the central point of the window
        horizontal_lines = []
        vertical_lines = []
        for ax1 in (u_ax, v_ax):
            horizontal_lines.append(ax1.axhline(i0, c='red', lw=0.25))
            vertical_lines.append(ax1.axvline(j0, c='red', lw=0.25))

        # updating function
        def on_click(event):

            # we want to make sure that we are actually clicking in the good axes
            if event.inaxes is None:
                return

            if event.inaxes != main_ax:
                # fetching the x and y position of the cursor
                j, i = int(event.xdata), int(event.ydata)

                # updating the main_ax image
                conv_im.set_data(self.convolutions[:, :, i, j])

                # updating the contours estimated from the polynomial surface
                # here this part is a bit tedious as we have to
                # compute the surface, remove and redraw the contours everytime
                # it doesn't seem to be too laggy however
                for tp in conv_cntrs[0].collections:
                    tp.remove()
                I, J = argmax2d(self.convolutions[:, :, i, j])
                xx, yy, z = poly.poly_surface(self.coeffs[i, j], I, J,
                                              degree=self.poly_degree,
                                              threshold=self.threshold)
                conv_cntrs[0] = main_ax.contour(xx - self.hlags.max(), yy - self.wlags.max(),
                                                z, colors='k', linewidths=0.5)

                # updating the horizontal and vertical lines and the moving windows
                for l1 in horizontal_lines:
                    l1.set_ydata([i, i])
                for l2 in vertical_lines:
                    l2.set_xdata([j, j])
                for c in patches:
                    c.center = j, i
                lx.set_xdata([self.shift[0, i, j], self.shift[0, i, j]])
                ly.set_ydata([self.shift[1, i, j], self.shift[1, i, j]])

            fig4.canvas.draw_idle()  # this is necessary so that it moves

        # this function ensures that we can keep clicking and moving
        def on_move(event):
            if event.button == 1:
                on_click(event)

        # connecting the figure to the different event functions
        fig4.canvas.mpl_connect('button_press_event', on_click)
        fig4.canvas.mpl_connect('motion_notify_event', on_move)

        # ensuring that zooming on one axes zooms on the others
        f_ax.get_shared_x_axes().join(g_ax, u_ax, v_ax)
        g_ax.get_shared_x_axes().join(f_ax, u_ax, v_ax)
        f_ax.get_shared_y_axes().join(g_ax, u_ax, v_ax)
        g_ax.get_shared_y_axes().join(f_ax, u_ax, v_ax)

        # ensuring that main_ax's limits do not change
        main_ax.set_xlim(main_ax.get_xlim())
        main_ax.set_ylim(main_ax.get_ylim())


if __name__ == '__main__':

    im0 = np.load('data/spongebob_warp2_0.npy', allow_pickle=True)
    im5 = np.load('data/spongebob_warp2_5.npy', allow_pickle=True)
    vf = np.load('data/vf_warp2.npy', allow_pickle=True) * 5

    path1 = 'data/3D_Base1.npy'
    path2 = 'data/3D_Monitor1.npy'
    data1 = np.load(path1)[:, :, 15]
    data2 = np.load(path2)[:, :, 15]

    std1 = (data1.max() - data1.min()) * 0.01
    std2 = (data2.max() - data2.min()) * 0.01

    noise1 = std1 * np.random.randn(data1.size).reshape(data1.shape)
    noise2 = std2 * np.random.randn(data2.size).reshape(data2.shape)

    data1 += noise1
    data2 += noise2

    maxlag = 20
    wlags = np.arange(-maxlag, maxlag + 1)
    hlags = np.arange(-maxlag, maxlag + 1)
    search_sigma = 10
    smooth_sigma = 12

    lccs = {}
    import time

    t0 = time.time()
    out = LCC2D(im5, im0, hlags, wlags,
                search_sigma, smooth_sigma,
                threshold=1)
    t1 = time.time()
    print(t1-t0)
    out.debug_plot()
    print()
    plt.show()
    """
    data1.dump('data/seismic_data1.npy')
    data2.dump('data/seismic_data2.npy')
    out.shift.dump('data/seismic_shift.npy')    
    """

    """
    # figure showing the x and y components respectively
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(8, 6))
    axs[0, 0].imshow(vf[0], origin='lower')
    axs[0, 0].text(0.01, 0.99, '$u_x$', ha='left', va='top', transform=axs[0, 0].transAxes)
    axs[0, 1].imshow(vf[1], origin='lower')
    axs[0, 1].text(0.01, 0.99, '$u_y$', ha='left', va='top', transform=axs[0, 1].transAxes)
    axs[1, 0].imshow(shift[0], vmin=vf[0].min(), vmax=vf[0].max(), origin='lower')
    axs[1, 0].text(0.01, 0.99, '$\hat{u}_x$', ha='left', va='top', transform=axs[1, 0].transAxes)
    axs[1, 1].imshow(shift[1], vmin=vf[1].min(), vmax=vf[1].max(), origin='lower')
    axs[1, 1].text(0.01, 0.99, '$\hat{u}_y$', ha='left', va='top', transform=axs[1, 1].transAxes)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    lccs[s] = out

    # figure showing the deformation field as a streamplot
    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10, 5))
    height, width = im0.shape
    H, W = np.arange(height), np.arange(width)
    ww, hh = np.meshgrid(W, H)
    norm = np.linalg.norm(vf, axis=0)
    axs[0].streamplot(ww, hh, vf[0], vf[1], color=norm, linewidth=0.5)
    norm2 = np.linalg.norm(shift, axis=0)
    norm2 = norm2 / norm.max()
    norm2[norm2 >= 1] = 1
    im = axs[1].streamplot(ww, hh, shift[0], shift[1], color=norm2, linewidth=0.5)

    axs[0].set_title('Original field', loc='left')
    axs[1].set_title('LCC approx.', loc='left')

    for ax in axs:
        ax.set_aspect(1)
        ax.set_ylim(0, height)
        ax.set_xlim(0, width)
    fig.subplots_adjust(wspace=0.05)
    """

    """
    shift = lcc.shift

    # figure showing the deformation field as a streamplot
    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10, 5))
    height, width = im0.shape
    H, W = np.arange(height), np.arange(width)
    ww, hh = np.meshgrid(W, H)
    norm = np.linalg.norm(vf, axis=0)
    axs[0].streamplot(ww, hh, vf[0], vf[1], color=norm, linewidth=0.5)
    norm2 = np.linalg.norm(shift, axis=0)
    norm2 = norm2 / norm.max()
    norm2[norm2 >= 1] = 1
    im = axs[1].streamplot(ww, hh, shift[0], shift[1], color=norm2, linewidth=0.5)

    axs[0].set_title('Original field', loc='left')
    axs[1].set_title('LCC approx.', loc='left')

    for ax in axs:
        ax.set_aspect(1)
        ax.set_ylim(0, height)
        ax.set_xlim(0, width)
    fig.subplots_adjust(wspace=0.05)

    # figure showing the x and y components respectively
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(8, 6))
    axs[0, 0].imshow(vf[0], origin='lower')
    axs[0, 0].text(0.01, 0.99, '$u_x$', ha='left', va='top', transform=axs[0, 0].transAxes)
    axs[0, 1].imshow(vf[1], origin='lower')
    axs[0, 1].text(0.01, 0.99, '$u_y$', ha='left', va='top', transform=axs[0, 1].transAxes)
    axs[1, 0].imshow(shift[0], vmin=vf[0].min(), vmax=vf[0].max(), origin='lower')
    axs[1, 0].text(0.01, 0.99, '$\hat{u}_x$', ha='left', va='top', transform=axs[1, 0].transAxes)
    axs[1, 1].imshow(shift[1], vmin=vf[1].min(), vmax=vf[1].max(), origin='lower')
    axs[1, 1].text(0.01, 0.99, '$\hat{u}_y$', ha='left', va='top', transform=axs[1, 1].transAxes)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    fig, ax = plt.subplots()
    norm = np.linalg.norm(shift, axis=0)
    nshift = shift/norm
    step = np.s_[::10, ::10]
    ax.quiver(ww[step], hh[step], nshift[0][step], nshift[1][step])
    """