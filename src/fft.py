import numpy as np
from scipy.fft import fft

tx = np.load("Data/210117_x_t_731.02_full.npy")
#f = fft(t)
l = len(tx)
tx = tx[int(0.525*l):int(0.53*l)]
ty = np.load("Data/210117_y_t_731.02_full.npy")
#f = fft(t)
ty = ty[int(0.525*l):int(0.53*l)]


np.save("Data/210117_x_t_731.02.npy", tx)
np.save("Data/210117_y_t_731.02.npy", ty)
