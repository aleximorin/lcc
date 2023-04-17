import numpy as np


def sinc_interp(t, x, t_out, axis=-1):
    T = t[1] - t[0]
    if len(x.shape) == 1:
        x = np.array(x)
    n = np.arange(x.shape[axis])

    out = np.add.outer(t_out, -n*T)/T
    out = np.sinc(out) @ x.T
    return out.T


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t_in = np.linspace(0, np.pi, 10)
    t_out = np.linspace(0, np.pi, 1000)

    nfreq = np.arange(1, 8, 2)

    x_in = np.sin(np.outer(nfreq, t_in))
    x_true = np.sin(np.outer(nfreq, t_out))
    x_out = sinc_interp(t_in, x_in, t_out)

    fig, axs = plt.subplots(len(nfreq), sharex='col', figsize=(8, 10))
    for i, n in enumerate(nfreq):
        axs[i].plot(t_out, x_true[i], label='True signal')
        axs[i].plot(t_in, x_in[i], label='Sampled signal', ls='dashed')
        axs[i].plot(t_out, x_out[i], label='Sinc interpolation')
        axs[i].text(0.99, 0.01, f'sin(${n} x$)', ha='right', va='bottom', transform=axs[i].transAxes)
    axs[0].legend(bbox_to_anchor=(0.5, 1), loc='lower center', frameon=False, ncol=3)
    fig.subplots_adjust(hspace=0)
    plt.show()


