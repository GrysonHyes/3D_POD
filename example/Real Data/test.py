from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def spatial(n_x, n_y, k, phi):
    k_image = phi[:,k].reshape(n_y, n_x)
    return k_image

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

if __name__ == '__main__':
    #--------- Params --------------
    image_pattern = "demo images/*.tif"
    #-------------------------------

    files = sorted(glob(image_pattern))

    if not files:
        raise FileNotFoundError("No file pattern exists.")

    print(f"Found {len(files)} images")

    # Import first image to get dimensions
    first = np.array(Image.open(files[0]), dtype=np.float32)
    n_y, n_x = first.shape
    n_t = len(files)

    # Initialize image array
    images = np.empty((n_t, n_y, n_x), dtype=np.float32)

    for i in range(0, n_t):
        images[i] = np.array(Image.open(files[i]), dtype=np.float32)

    x = np.arange(n_x, dtype=float)
    y = np.arange(n_y, dtype=float)
    t = np.arange(n_t, dtype=float)

    S = images.reshape(n_t, -1).T

    lamb, phi, a = pod(S)
    print(lamb.shape)

    for k in range(1,4):
        img = spatial(n_x, n_y, k, phi)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            img,
            cmap='RdBu_r',           # diverging colormap centered at 0
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'POD Mode {k}')
        plt.colorbar(im, ax=ax, label='mode amplitude')
        plt.show()

    modes = np.arange(0, n_t, 1)

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(modes, lamb)
    ax1.set_ylabel('Energy Content')
    ax1.set_xlabel('Mode Number')
    ax1.set_xlim((0, 10))

    TKE_k = lamb / np.sum(lamb)
    TKE_cum = np.cumsum(TKE_k)

    ax2.plot(modes, TKE_cum)
    ax2.set_ylabel('Energy Fraction')
    ax2.set_xlim((0, 10))
    plt.show()
