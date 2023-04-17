import numpy as np
import matplotlib.pyplot as plt
import gstools as gs


def move_points(X_points, X_V, V, dt, N):
    dim = len(X_points)
    npoints = len(X_points.T)
    if dim > 2:
        raise ValueError('3D not implemented yet')

    # we generate an interpolating function for vx and vy
    from scipy.interpolate import RectBivariateSpline
    fx = RectBivariateSpline(np.unique(X_V[0]), np.unique(X_V[0]), V[0])
    fy = RectBivariateSpline(np.unique(X_V[1]), np.unique(X_V[1]), V[1])

    X_points = X_points.copy()

    X_out = np.zeros((N, dim, npoints))
    for i in range(N):
        X_points[0] += fx(*X_points, grid=False)*dt
        X_points[1] += fy(*X_points, grid=False)*dt

    return X_points


def generate_constant_field(X, vx, vy):
    V = np.zeros_like(X)
    V[0] = vx
    V[1] = vy
    return V


def rasterize(points, x):
    H, _, _ = np.histogram2d(*points, (x, x))
    return H


def generate_random_field(X, var=1, len_scale=10, seed=125051213, full=False):
    model = gs.Gaussian(dim=len(X), var=var, len_scale=len_scale)
    srf = gs.SRF(model, generator="VectorField", seed=seed)  # Spatial Random Field
    V = srf(X)  # can only take in a grid?
    if full:
        return V, srf
    else:
        return V


class RandomVelocityField:

    def __init__(self, X, var=1, len_scale=10, seed=None, interpolate_step=None):

        self.X = X
        self.dim = len(self.X)
        self.nd = self.X.shape[1:]

        self.var = var
        self.len_scale = len_scale
        self.seed = seed
        self.srf = self._generate_random_field()

        if interpolate_step is not None:
            self.V = self._interpolate_field(interpolate_step)
        else:
            self.V = self.srf(self.X).reshape(self.X.shape)

    def _generate_random_field(self):
        model = gs.Gaussian(dim=self.dim, var=self.var, len_scale=self.len_scale)
        return gs.SRF(model, generator="VectorField", seed=self.seed)  # Spatial Random Field

    def _interpolate_field(self, step):
        x, y = self.X
        subx = x[::step, ::step]
        suby = y[::step, ::step]
        subv = self.srf((subx, suby)).reshape(2, *subx.shape)

        from scipy.interpolate import RectBivariateSpline
        fx = RectBivariateSpline(subx[0], suby[:, 0], subv[0].T)
        fy = RectBivariateSpline(subx[0], suby[:, 0], subv[1].T)

        Vx = fx(self.X[0].flatten(), self.X[1].flatten(), grid=False).reshape(self.X[0].shape)
        Vy = fy(self.X[0].flatten(), self.X[1].flatten(), grid=False).reshape(self.X[0].shape)
        V = np.array((Vx, Vy))

        return V

    def plot(self, ax=None):
        if self.dim > 2:
            raise ValueError('3D not implemented yet')

        norm = np.linalg.norm(self.V, axis=0)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        else:
            fig = plt.gcf()
        ax.streamplot(self.X[0], self.X[1], self.V[0], self.V[1], linewidth=norm, color=norm)
        # ax.contourf(self.X[0], self.X[1], norm)

        return fig, ax

    def move_points(self, X_points, dt, ntimes):

        dim = len(X_points)
        npoints = len(X_points.T)

        if dim > 2:
            raise ValueError('3D not implemented yet')
        if dim != self.dim:
            raise ValueError('Dimensions of X_points and self.X don\'t match')

        # we generate an interpolating function for vx and vy
        from scipy.interpolate import RectBivariateSpline
        fx = RectBivariateSpline(np.unique(self.X[0]), np.unique(self.X[1]), self.V[0])
        fy = RectBivariateSpline(np.unique(self.X[0]), np.unique(self.X[1]), self.V[1])

        X_out = np.zeros((ntimes + 1, dim, npoints))
        X_out[0] = X_points

        for i in range(1, ntimes + 1):
            X_out[i, 0] = X_out[i-1, 0] + fx(*X_out[i-1], grid=False) * dt
            X_out[i, 1] = X_out[i-1, 1] + fy(*X_out[i-1], grid=False) * dt

        return X_out


if __name__ == '__main__':
    # coordinate generation
    n = 100
    x = np.linspace(0, 100, n)
    X = np.array(np.meshgrid(x, x))

    # custom function to generate constant field
    Vc = generate_constant_field(X, 3, 1)

    # gstools can generate a random vector field
    vf = RandomVelocityField(X, var=5, len_scale=15, seed=42069)
    vf.V[0] = 5
    vf.V[1] = 2

    vf.plot()

    # putting points on the field and moving them, rasterizing them after
    npoints = 10000
    offset = 0.0 * np.ptp(x)
    points = np.random.uniform(x.min() + offset, x.max() - offset, size=(2, npoints))

    dt = 0.001
    nmove = 5000
    points2 = vf.move_points(points, dt, nmove)

    #plt.plot(points2[:, 0], points2[:, 1])

    fig, axs = plt.subplots(1, 2)
    h1 = rasterize(points2[nmove//2], x).T
    h2 = rasterize(points2[-1], x).T
    axs[0].imshow(h1, extent=[x.min(), x.max(), x.min(), x.max()], origin='lower')
    axs[1].imshow(h2, extent=[x.min(), x.max(), x.min(), x.max()], origin='lower')

    np.save('x.npy', x)
    np.save('h1.npy', h1)
    np.save('h2.npy', h2)

    plt.show()
    print()
