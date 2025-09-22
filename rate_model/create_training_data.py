#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from tqdm import tqdm
import xarray as xr

from rate_model import RateModelWC


SEED = 113
np.random.seed(SEED)

# Model params
N = 2
w_sigma = 0.5
h0_sigma = 1
model_par = dict(tau=1, rmax=10, gain_slope=0.5, gain_center=0)
sim_par = {'dt': 0.25, 'nsteps': 100}

# Dataset params
n_model_gens = 512
n_point_gens = 32

# Create xarray dataset
X = xr.Dataset(
    {
        'W': (['model', 'pop_in', 'pop'], np.zeros((n_model_gens, N, N))),
        'H': (['model', 'point', 'pop'], np.zeros((n_model_gens, n_point_gens, N))),
        'R': (['model', 'point', 'pop'], np.zeros((n_model_gens, n_point_gens, N))),
    },
    coords={
        'model': np.arange(n_model_gens),
        'point': np.arange(n_point_gens),
        'pop': np.arange(N),
        'pop_in': np.arange(N)
    }
)
""" X.attrs = {
    'seed': SEED,
    'w_sigma': w_sigma,
    'h0_sigma': h0_sigma,
    **model_par, **sim_par
} """


#for n in tqdm(range(n_model_gens)):
for n in range(n_model_gens):
    print(f'Model: {n + 1}/{n_model_gens}', flush=True)

    # Generate weights
    #W = w_sigma * np.random.randn(N, N)
    W = w_sigma * (2 * np.random.rand(N, N) - 1)
    X['W'].loc[{'model': n}] = W

    # Initialize model
    model = RateModelWC(W, **model_par)

    for m in (range(n_point_gens)):
        # Generate external input
        #h0 = h0_sigma * np.random.randn(N, 1)
        h0 = h0_sigma * (2 * np.random.rand(N, 1) - 1)
        X['H'].loc[{'model': n, 'point': m}] = h0.ravel()

        # Run the model
        X['R'].loc[{'model': n, 'point': m}] = model.run(
            h0, r0=np.zeros((N, 1)), **sim_par).ravel()

# Save the dataset to a NetCDF file
dt, nsteps = sim_par['dt'], sim_par['nsteps']
fname_out = (f'rates_npops_{N}_nmodels_{n_model_gens}_npts_{n_point_gens}_seed_{SEED}'
             f'_ws_{w_sigma}_hs_{h0_sigma}_uni_dt_{dt}_nt_{nsteps}.nc')
X.to_netcdf(f'data/{fname_out}')

#data = xr.open_dataset('data/training_data.nc')
#print(data)
#data.close()
