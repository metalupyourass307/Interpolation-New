import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift

df = pd.read_csv('/Users/laiyuxuan/Desktop/Interpolation/H/mapped_coordinates.csv')
x_arr = df['x'].values
y_arr = df['y'].values

def rho_of_r(x, y):
    r = np.sqrt(x**2 + y**2)
    rho = (1 / np.pi) * np.exp(-2 * r)
    return rho

x_grid = np.linspace(-1.50 , 1.4875 , 64)
y_grid = np.linspace(-1.50 , 1.4875, 64)

x_vals, y_vals = np.meshgrid(x_grid, y_grid)
rho_arr = np.zeros_like(x_vals)
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        rho_arr[i, j] = rho_of_r(x_vals[i, j], y_vals[i, j])

rho_grid = rho_arr.reshape((64, 64))

ck = fft2(rho_grid, norm='ortho')
ck_shifted = fftshift(ck)
def fourier_interp_2d(ck, x_vals, y_vals):
    N = ck.shape[0]
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=2/N)) 
    ku = k[:, None] * x_vals[None, :]  
    kv = k[None, :] * y_vals[:, None] 
    phase = np.exp(1j * (ku[:, :, None] + kv[None, :, :])) 
    rho_interp = np.sum(ck_shifted[:, :, None, None] * phase, axis=(0, 1))
    return rho_interp.real

rho_interp = fourier_interp_2d(ck, x_grid, y_grid)


y_vals_line = y_arr.reshape((64,64))[32, :]
sorted_idx = np.argsort(y_vals_line)
y_sorted = y_vals_line[sorted_idx]
x_index = len(rho_interp) // 2

plt.plot(y_sorted, rho_interp[x_index, :], label='Reconstructed ρ(y)', color='red', linestyle='--')
plt.xlim(-3,3)
plt.xlabel('y')
plt.ylabel('ρ(y)')
plt.title('Reconstructed Density along y (x = center)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
