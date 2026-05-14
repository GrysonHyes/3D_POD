'''
Code used for 3-dimensional (2 spatial dimensions) POD.
Author: Grayson S. Hayes
Corresponding Email: hayes.grayson@outlook.com
'''

import numpy as np
import matplotlib.pyplot as plt

def load_npz(data, field_key = 'psi', t_key = 't', x_key = 'x', y_key = 'y'):
    '''Load 2 spatial dimension field stored as (n_t, n_y, n_x)'''
    d = np.load(data)
    field = d[field_key]
    n_t = field.shape[0]
    S = field.reshape(n_t, -1).T
    grid = {
        'x' : d[x_key],
        'y' : d[y_key]
    }

    return S, d[t_key], grid

def load_csv(data, time_column = 0, headers = False):
    '''Load a CSV assuming that each row is a time step
    First column should be time and remaining columns should be spatial data'''
    skip = 1 if headers else 0
    d = np.loadtxt(data, delimiter=',', skiprows=skip)
    t = d[:, time_column]
    space = [i for i in range(d.shape[1]) if i != time_column]
    S = d[:, space].T
    return S, t, None

def pod(S: np.array, method: str = 'snapshot') -> tuple[np.array, np.array, np.array]:
    S_fluc = S - S.mean(axis = 1, keepdims = True)

    if method == 'snapshot':
        m = S.shape[1]
        C = (m-1)**-1 * (S_fluc.T @ S_fluc)
        lamb, phi = np.linalg.eigh(C)
        lamb = lamb[::-1] # Sorts eigenvalues in descending order
        phi = phi[:,::-1]
        phi = S_fluc @ phi # Comment if you want temporal modes
        phi /= np.linalg.norm(phi, axis=0, keepdims=True) # Comment if you want temporal modes

    elif method == 'full':
        m = S.shape[0]
        C = (m-1)**-1 * (S_fluc @ S_fluc.T)
        lamb, phi = np.linalg.eigh(C)
        lamb = lamb[::-1] # Sorts eigenvalues in descending order
        phi = phi[:,::-1]

    else:
        raise ValueError(f"Method must be 'snapshot' or 'full', got {method}")

    a = phi.T @ S_fluc
    
    return lamb, phi, a

def spatial(n_y, n_x, k, phi):
    k_image = phi[:,k].reshape(n_y, n_x)

    return k_image

'''
DEMO
'''

if __name__ == '__main__':

    data = 'example/Synthetic Data/sample.npz'

    d = np.load(data)
    psi, x, y, t = d['psi'], d['x'], d['y'], d['t']
    nx = len(x)
    ny = len(y)

    # Pick a time index
    k = 1  # first snapshot; try 25, 50, 75 to see the orbit progress

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        psi[k],
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        cmap='viridis',          
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(f'Streamfunction at t = {t[k]:.3f}')
    plt.colorbar(im, ax=ax, label=r'$\psi$')
    plt.show()

    S, t, grid = load_npz(data)
    lamb, phi, a = pod(S, method='full')

    TKE_k = lamb / np.sum(lamb)

    modes = np.arange(1, len(lamb) + 1)

    fig, ax1 = plt.subplots()
    ax1.semilogy(modes, TKE_k, label = 'Percentage of Energy', color = 'r')
    ax1.set_xlabel('Mode #')
    ax1.set_ylabel('TKE%')
    ax1.tick_params(axis='y', color='r')

    ax2 = ax1.twinx()
    TKE_cum = np.cumsum(TKE_k)
    ax2.plot(modes, TKE_cum, label = 'Cumulative Energy', color = 'b')
    ax2.set_ylabel('Cumulative Energy Fraction')
    ax2.tick_params(axis='y', color='b')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax1.set_xlim((0,10)) # Feel free to mess with this
    plt.title('Energy Content of POD Modes')
    plt.show()

    for k in range(1,4):
        img = spatial(ny, nx, k, phi)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            img,
            extent=[grid['x'][0], grid['x'][-1], grid['y'][0], grid['y'][-1]],
            origin='lower',
            cmap='RdBu_r', 
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'POD Mode {k}')
        plt.colorbar(im, ax=ax, label='mode amplitude')
        plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for k, ax in enumerate(axes):
        mode_image = spatial(ny, nx, k, phi)
        # Symmetric color limits centered at 0
        vmax = np.abs(mode_image).max()
        im = ax.imshow(
            mode_image,
            extent=[grid['x'][0], grid['x'][-1], grid['y'][0], grid['y'][-1]],
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Mode {k+1}, λ={lamb[k]:.3g}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
