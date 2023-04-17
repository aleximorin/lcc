import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np


if __name__ == '__main__':

    path1 = 'data/3D_Base1.npy'
    path2 = 'data/3D_Monitor1.npy'
    data1 = np.load(path1)
    data2 = np.load(path2)

    cmap = plt.cm.get_cmap("viridis", 4)

    grid = pv.UniformGrid(spacing=(1, 5, 25), dimensions=np.array(data1.shape) + 1)
    vmin = np.quantile(data1, 0.0225)
    vmax = np.quantile(data1, 0.0975)
    # data1[(data1 > vmin) & (data1 < vmax)] = np.nan
    grid.cell_data['value'] = data1.flatten(order='F')

    # grid.plot(cmap=cmap, opacity='sigmoid', volume=True)

    p = pv.Plotter()
    p.add_mesh_clip_plane(grid, cmap=cmap)
    p.show()

