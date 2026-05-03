'''
Library Functions - Importable from anywhere
'''

import numpy as np

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

def pod(S, method = 'snapshot'):
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

def spatial(n_x, n_y, k, phi):
    k_image = phi[:,k].reshape(n_y, n_x)
    return k_image