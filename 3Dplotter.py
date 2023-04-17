import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

dimensions = {'x': 0, 'y': 1, 'z': 2}
cmap = plt.cm.get_cmap('seismic')
absolute_cmap = plt.cm.get_cmap('gray')


def normalize(x, axis=-1):

    if axis == -1:
        axis = len(x.shape) - 1

    minima = x.min(axis=axis)
    maxima = x.max(axis=axis)

    minima = minima.reshape(np.insert(minima.shape, axis, 1))
    maxima = maxima.reshape(np.insert(maxima.shape, axis, 1))

    tmp = (x - minima)/(maxima - minima)
    return tmp


class DataSlice:

    def __init__(self, axis, slicer):
        self.axis = axis
        self.index = dimensions[axis]
        self.origin = [0.5, 0.5, 0.5]
        self.mesh = slicer.mesh

        self.output = self.mesh.slice(normal=self.axis, origin=self.origin)
        slicer.plotter.add_mesh(self.output, cmap=cmap, scalars='amplitude', clim=(slicer.vmin, slicer.vmax))

    def __call__(self, value):

        self.origin[self.index] = value
        self.update()

    def update(self):
        tmp = self.mesh.slice(normal=self.axis, origin=self.origin)
        self.output.copy_from(tmp)
        return


class CubeSlicer:

    def __init__(self, data, threshold=0.975):

        # here we define the data on a mesh object
        self.data = data
        h, w, d = self.data.shape

        self.mesh = pv.UniformGrid(spacing=(1 / h, 1 / w, 1 / d),
                                   dimensions=np.array(data.shape) + 1)
        self.mesh.cell_data['amplitude'] = data.flatten(order='F')

        absolute_amplitude = np.abs(data)
        self.mesh.cell_data['absolute_amplitude'] = absolute_amplitude.flatten(order='F')

        self.vmin = np.nanquantile(self.data, 1 - threshold)
        self.vmax = np.nanquantile(self.data, threshold)

        self.plotter = pv.Plotter()
        self.plotter.add_mesh_slice_orthogonal(self.mesh, cmap=cmap, clim=(self.vmin, self.vmax))
        #self.plotter.add_mesh(self.mesh.outline(), color='k')

        #self.plotter.add_mesh(self.mesh, scalars='absolute_amplitude', opacity='sigmoid', cmap=absolute_cmap)
        #self.plotter.add_mesh_threshold(self.mesh, scalars='absolute_amplitude', cmap=absolute_cmap)

        """self.sliceX = DataSlice('x', self)
        self.sliceY = DataSlice('y', self)
        self.sliceZ = DataSlice('z', self)

        self.plotter.add_slider_widget(callback=lambda value: self.sliceX(value),
                                       rng=[1e-9, 1],
                                       value=0.5,
                                       fmt='x=%.2f',
                                       pointa=(0.025, 0.3),
                                       pointb=(0.31, 0.3),
                                       style='modern',
                                       interaction_event='always')

        self.plotter.add_slider_widget(callback=lambda value: self.sliceY(value),
                                       rng=[1e-9, 1],
                                       value=0.5,
                                       fmt='y=%.2f',
                                       pointa=(0.025, 0.2),
                                       pointb=(0.31, 0.2),
                                       style='modern',
                                       interaction_event='always')

        self.plotter.add_slider_widget(callback=lambda value: self.sliceZ(value),
                                       rng=[1e-9, 1],
                                       value=0.5,
                                       fmt='z=%.2f',
                                       pointa=(0.025, 0.1),
                                       pointb=(0.31, 0.1),
                                       style='modern',
                                       interaction_event='always')"""
        self.plotter.show()


if __name__ == '__main__':
    from scipy.io import loadmat
    path1 = 'data/Findelen_2207_datacube_700_2.mat'
    path2 = 'data/Findelen_2210_datacube_700_2.mat'

    cube1 = loadmat(path1, mat_dtype=True, squeeze_me=True, struct_as_record=False)
    cube2 = loadmat(path2, mat_dtype=True, squeeze_me=True, struct_as_record=False)

    data1 = cube1['datacube_2'].datacube
    data2 = cube2['datacube_2'].datacube

    plt.close('all')
    line = 270
    im1 = data1[:, line]
    im2 = data2[:, line]
    x = cube1['datacube_2'].x
    y = cube1['datacube_2'].y
    t = cube1['datacube_2'].t
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, data1[-1])
    ax.axhline(y[line])

    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all')
    axs[0].pcolormesh(x, t, im1)
    axs[1].pcolormesh(x, t, im2)
    axs[1].invert_yaxis()
    plt.show()

    im1.dump('data/gpr_slice1.npy')
    im2.dump('data/gpr_slice2.npy')

    #CubeSlicer(data1, threshold=0.85)
