import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import root, root_scalar

import torch


def polymatrix2d(x, y, order=2):
    ij = itertools.product(range(order + 1), range(order + 1))

    # z needs to be of shape (3*3, height * width)
    size = len(x)

    ncols = (order + 1) ** 2
    G = torch.zeros((size, ncols))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j

    return G


def vec_polyfit2d(x, y, z, order=2):
    G = polymatrix2d(x, y, order=order)
    m, _, _, _ = torch.linalg.lstsq(G, z, rcond=1e-10)
    return m


class PytorchPolyFit2D(torch.nn.Module):
    def __init__(self, X, images):
        super(PytorchPolyFit2D, self).__init__()

        self.x, self.y = X
        self.images = images

        self.X0 = torch.nn.Parameter(torch.zeros(2, self.images.shape[-1]))
        self.coeffs = vec_polyfit2d(self.x, self.y, self.images)

    def forward(self):
        G = polymatrix2d(*self.X0)
        return torch.einsum('ij, ji -> i', G, self.coeffs)

    def fit(self, n=800, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        out = np.nan
        for i in range(n):
            print(f'\riteration {i+1}/{n} = {(i+1) / n * 100:.2f}%, loss = {out:.2f}', end='')
            out = torch.norm(1 - self.forward())
            out.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.X0.detach()

def polyfit2d(x, y, z, order=3, product=False):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent

    if product:
        ij = itertools.product(range(order+1), range(order+1))
    else:
        ij = np.vstack((np.arange(order + 1), np.zeros(order + 1)))
        ij = np.hstack((ij, np.roll(ij[:, 1:], 1, axis=0))).T

    ncols = (order + 1)**2 if product else len(ij)
    G = np.zeros((x.size, ncols))

    for k, (i,j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=1e-10)
    return m


def polyval2d(x, y, m, order, product=True):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    if product:
        ij = itertools.product(range(order+1), range(order+1))
    else:
        ij = np.vstack((np.arange(order + 1), np.zeros(order + 1)))
        ij = np.hstack((ij, np.roll(ij[:, 1:], 1, axis=0))).T

    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z = z + a * x**i * y**j
    return z


def poly_interp(corr, I, J, threshold=1):
    height, width = corr.shape
    h, w = np.arange(height), np.arange(width)
    xx, yy = np.meshgrid(h, w)

    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)

    m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=2, product=False)
    Ihat = -0.5 * m[1] / m[2]
    Jhat = -0.5 * m[3] / m[4]

    return Jhat, Ihat


def poly_interp_newton(corr, I, J, degree=2, threshold=1):

    # we define our polynomial f as:
    # f(x, y)=m[0]+m[1]*y+m[2]*yy+m[3]*x+m[4]*x*y+m[5]*x*yy+m[6]*xx+m[7]*xx*y+m[8]*xx*yy
    # df(x, y)/dx = m[3] + m[4] * y + m[5] * yy + m[6] * 2x + m[7] * 2x * y + m[8] * 2x * yy
    # df(x, y)/dy = m[1] + m[2] * 2y + m[4] * x + m[5] * x * 2y + m[7] * xx + m[8] * xx * 2y

    height, width = corr.shape
    h, w = np.arange(height), np.arange(width)
    xx, yy = np.meshgrid(h, w)

    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)
    m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=degree, product=True)

    dfdx = lambda X: m[3] + m[4] * X[1] + m[5] * X[1] * X[1] + m[6] * 2 * X[0] + \
                     m[7] * 2 * X[0] * X[1] + m[8] * 2 * X[0] * X[1] * X[1]
    dfdy = lambda X: m[1] + m[2] * 2*X[1] + m[4] * X[0] + m[5] * X[0] * 2 * X[1] + \
                     m[7] * X[0] * X[0] + m[8] * X[0] * X[0] * 2 * X[1]

    grad = lambda X: np.array([dfdx(X), dfdy(X)])

    j0, i0 = root(grad, x0=np.array([J, I])).x

    return i0, j0, m


def poly_interp_library(corr, I, J, degree=2, threshold=2):
    height, width = corr.shape
    h, w = np.arange(height), np.arange(width)
    xx, yy = np.meshgrid(h, w)

    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)
    poly = ndPolynomial(degree=degree)
    X = np.array((xx[ii], yy[ii]))
    poly.fit(X, corr[ii])

    j0, i0 = poly.optimum([J, I])

    return i0, j0, poly.m


def poly_surface(m, I, J, degree=2, threshold=1, N=25):
    x = np.linspace(J - threshold - 0.5, J + threshold + 0.5, N)
    y = np.linspace(I - threshold - 0.5, I + threshold + 0.5, N)
    xx, yy = np.meshgrid(x, y)
    z = polyval2d(xx.flatten(), yy.flatten(), m, order=degree).reshape(xx.shape)
    return xx, yy, z


class ndPolynomial:

    # an attempt at vectorized evaluation of a nd polynomial and it's 1st derivative

    def __init__(self, degree, m=None, powers=None, build_grad=True):
        self.degree = degree
        self.m = m
        self.nd = None
        self.nm = None

        self.grad = None
        self.powers = powers

        # manual lock to not recursively build the gradient?
        self.build_grad = build_grad

    def fit(self, X, z):

        X = self._assert_nd(X)

        self.nd, ns = X.shape
        self.nm = (self.degree + 1) ** self.nd

        G = self._G(X)
        self.m, _, _, _ = np.linalg.lstsq(G, z, rcond=1e-10)

        if self.build_grad:
            self._build_grad()

    def optimum(self, x0):

        x0 = self._assert_nd(x0)

        from scipy.optimize import root
        x0 = root(self.grad, x0=x0, args={'optimum': True}).x
        return x0

    def _assert_nd(self, X):

        if np.isscalar(X):
            X = np.array([X])

        if isinstance(X, list):
            if len(X) == self.nd:
                X = np.array(X)[:, None]
            else:
                raise ValueError('Dimensions of X are inappropriate. X should be of shape (n_dimension, n_obs)')

        if len(X.shape) == 1:
            if self.nd is not None:
                if self.nd == 1:
                    X = np.array(X)[None, :]
                else:
                    assert len(X) == self.nd, 'If inputting a list, length of X should be the same as the number of dimensions'
                    X = np.array(X)[:, None]

        return X.astype(float)

    def _build_grad(self):

        polynomials = []
        ijk = self._ijk()
        for i in range(self.nd):
            derivative = ijk.copy()
            derivative[:, i] -= 1
            derivative[derivative < 0] = 0  # we ensure that we are not raising zero to negative powers
            m = self.m * ijk[:, i]

            poly = ndPolynomial(degree=self.degree-1,
                                m=m,
                                powers=derivative,
                                build_grad=False)
            poly.nd = self.nd
            poly.nm = self.nm
            polynomials.append(poly)

        def grad(X, optimum=False):
            out = np.array([poly(X) for poly in polynomials])

            if optimum:
                out = out[:, 0]

            return out

        self.grad = grad

    def __call__(self, X):
        # we can simply build a corresponding G matrix and multiply it by our coefficient vectors
        G = self._G(X)
        return G @ self.m

    def _ijk(self):
        if self.powers is None:
            ijk = itertools.product(*[range(self.degree + 1) for _ in range(self.nd)])
            ijk = np.array([i for i in ijk])
            return ijk
        return self.powers

    def _G(self, X):
        X = self._assert_nd(X)

        _, ns = X.shape
        ijk = self._ijk()

        #  vectorized construction of the G matrix used for polynomial evaluation
        G = np.stack([X for _ in range(self.nm)])
        G = np.prod(G ** ijk[:, :, None], axis=1)
        return G.T


if __name__ == '__main__':

    x = np.linspace(-1000, 1000, 25)
    z = 5 + 2*x + 0.1 * x * x

    """# 1d example
    poly = ndPolynomial(degree=2)
    poly.fit(x, z)
    zhat = poly(x)
    poly.optimum(10)

    plt.figure()
    plt.scatter(x, z)
    plt.plot(x, zhat)
    plt.title('1d example')"""

    # 2d example
    poly2d = ndPolynomial(degree=2)
    xx, yy = np.meshgrid(x, x)
    X = np.array((xx.flatten(), yy.flatten()))
    Z = 1 + 0.1*X[0] + 0.3*X[0]**2 + 0.6*X[1] + 0.2*X[1]**2 + 0.1*X[0]*X[1] + \
        0.05*X[1]*(X[0]**2) + 0.08*X[0]*(X[1]**2) + 0.01*(X[0]**2)*(X[1]**2)

    poly2d.fit(X, Z)
    Zhat = poly2d(X)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #ax.scatter(X[0], X[1], Z)
    x0 = poly2d.optimum([10, 10])
    ax.scatter(*x0, poly2d(x0))
    print(x0)
    ax.plot_surface(xx, yy, Zhat.reshape(xx.shape), alpha=0.5)
    ax.set_title('2d example')

    plt.show()