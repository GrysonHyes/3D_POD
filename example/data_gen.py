'''
Code used for dataset generation. Used for debugging and testing. Vibecoded a little tbh.
'''

import numpy as np
 
 
# ---------- parameters ----------
# Domain
x_min, x_max = 0.0, 2.0
y_min, y_max = 0.0, 2.0
n_x, n_y = 80, 80          # spatial resolution
 
# Time
t_min, t_max = 0.0, 2.0 * np.pi
n_t = 100                 # number of snapshots (covers one full orbit)
 
# Vortex parameters
x_c, y_c = 1.0, 1.0        # orbit center
R = 0.4                    # orbit radius
r = 0.2                    # Gaussian core size
omega = 1.0                # angular orbital speed
 
# Noise (set to 0 for clean data)
noise_level = 0.0
seed = 0
 
output_file = "sample.npz"
# ---------------------------------
 
 
# Spatial grid
x = np.linspace(x_min, x_max, n_x)
y = np.linspace(y_min, y_max, n_y)
X, Y = np.meshgrid(x, y, indexing="xy")   # shape (n_y, n_x)
 
# Time grid
t = np.linspace(t_min, t_max, n_t)
 
# Allocate field array: (n_t, n_y, n_x)
psi = np.empty((n_t, n_y, n_x))
 
for k, tk in enumerate(t):
    # Vortex centers at this time
    x1 = x_c + R * np.cos(omega * tk)
    y1 = y_c + R * np.sin(omega * tk)
    x2 = x_c - R * np.cos(omega * tk)
    y2 = y_c - R * np.sin(omega * tk)
 
    g1 = np.exp(-((X - x1) ** 2 + (Y - y1) ** 2) / (2 * r ** 2))
    g2 = np.exp(-((X - x2) ** 2 + (Y - y2) ** 2) / (2 * r ** 2))
    psi[k] = g1 + g2
 
# Optional noise
if noise_level > 0:
    rng = np.random.default_rng(seed)
    psi = psi + noise_level * rng.standard_normal(psi.shape)
 
np.savez(output_file, psi=psi, x=x, y=y, t=t)
 
print(f"Saved field to {output_file}")
print(f"  psi shape: {psi.shape}   (n_t, n_y, n_x)")
print(f"  grid: {n_x} x {n_y} = {n_x * n_y} spatial points")
print(f"  time: {n_t} snapshots from {t_min:.3f} to {t_max:.3f}")
print(f"  file size on disk: ~{psi.nbytes / 1e6:.1f} MB")
