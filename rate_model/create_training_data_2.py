#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from tqdm import tqdm
import xarray as xr

from rate_model import RateModelWC


SEED = 112
np.random.seed(SEED)

# Model params
N = 2
w_sigma = 0.5
h0_sigma = 1
model_par = dict(tau=1, rmax=10, gain_slope=0.5, gain_center=0)
sim_par = {'dt': 0.5, 'nsteps': 20}

# Dataset params
n_model_gens = 1
n_point_gens = 64

# Create xarray dataset
X = xr.Dataset(
    {
        'W': (['model', 'pop_in', 'pop'], np.zeros((n_model_gens, N, N))),
        'H': (['model', 'point', 'pop'], np.zeros((n_model_gens, n_point_gens, N))),
        'R': (['model', 'point', 'pop', 'time'],
              np.zeros((n_model_gens, n_point_gens, N, sim_par['nsteps']))),
        'DR': (['model', 'point', 'pop'], np.zeros((n_model_gens, n_point_gens, N))),
    },
    coords={
        'model': np.arange(n_model_gens),
        'point': np.arange(n_point_gens),
        'pop': np.arange(N),
        'pop_in': np.arange(N),
        'time': np.arange(sim_par['nsteps'])
    }
)

for n in range(n_model_gens):
    print(f'Model: {n + 1}/{n_model_gens}', flush=True)

    # Generate weights
    W = w_sigma * np.random.randn(N, N)
    X['W'].loc[{'model': n}] = W

    # Initialize model
    model = RateModelWC(W, **model_par)
    model.sim_res_type = 'full'

    for m in range(n_point_gens):
        cc = {'model': n, 'point': m}

        # Generate external input
        h0 = h0_sigma * np.random.randn(N, 1)
        X['H'].loc[cc] = h0.ravel()

        # Run the model
        R = model.run(h0, r0=np.zeros((N, 1)), **sim_par)
        X['R'].loc[cc] = R.values.ravel()
        DR = R.isel(time=-1, drop=True) - R.isel(time=1, drop=True)
        X['DR'].loc[cc] = DR.values.ravel()

# Compute statistics for unrolled absolute values of X['DR']
#dr_abs = np.abs(X['DR'].values.ravel())
dr_abs = np.abs((X['DR'] / X['R']).values.ravel())
print(f"Min: {dr_abs.min():.04f}, Max: {dr_abs.max():.04f}, "
      f"Mean: {dr_abs.mean():.04f}, Std: {dr_abs.std():.04f}")

import matplotlib.pyplot as plt

# Plot histogram of dr_abs
plt.hist(dr_abs, bins=50, color='blue', alpha=0.7,
         range=(0, 0.03))
plt.title('Histogram of |DR|')
plt.xlabel('|DR|')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Save the dataset to a NetCDF file
#fname_out = (f'rates_npops_{N}_nmodels_{n_model_gens}_npts_{n_point_gens}_seed_{SEED}'
#             f'_ws_{w_sigma}_hs_{h0_sigma}.nc')
#X.to_netcdf(f'data/{fname_out}')

#data = xr.open_dataset('data/training_data.nc')
#print(data)
#data.close()
