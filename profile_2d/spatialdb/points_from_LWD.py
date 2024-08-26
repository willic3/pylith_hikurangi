#!/usr/bin/env nemesis

"""
Python script to get points to add to spatial database from Logging-While-Drilling data.

Do this by creating layers that follow topography and interpolating along those layers
between borehole points for vp, vs, and density.


"""

import pandas as pd
import numpy as np
import math
import sys
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import h5py

## -------- Load in topopgraphy points --------- ##

topo = np.loadtxt("/Users/mckenziecarlson/Documents/GitHub/pylith_hikurangi/profile_2d/mesh/groundsurf-profile-coords2d.tsv")

p_U1518 = [-5541.6171, -2849.1, 0.0]
p_U1519 = [-34030.0862, -1264.0, 0.0]

## ------ Load in LWD Data and remove nans ----- ##

# U1518 ----------
U1518_vel = pd.read_csv('/Users/mckenziecarlson/pylith/pylith-4.1.1-macOS-10.15-x86_64/aotearoa/U1518_vpvs.csv',skiprows=5)
U1518_vel.rename(columns={'m': 'depth(m)', 'km/s': 'vp(km/s)', 'km/s.1': 'vs(km/s)'}, inplace=True)
U1518_vel[U1518_vel==(-999.25)]=np.nan
U1518_vel = U1518_vel.sort_values('depth(m)')

U1518_den = pd.read_csv('/Users/mckenziecarlson/pylith/pylith-4.1.1-macOS-10.15-x86_64/aotearoa/U1518_density.csv',skiprows=5)
U1518_den.rename(columns={'m': 'depth(m)', 'g/cm3': 'density(g/cm3)'}, inplace=True)
U1518_den[U1518_den==(-999.25)]=np.nan
U1518_den = U1518_den.sort_values('depth(m)')


#U1519 -----------
U1519_vel = pd.read_csv('/Users/mckenziecarlson/pylith/pylith-4.1.1-macOS-10.15-x86_64/aotearoa/U1519_vpvs.csv',skiprows=5)
U1519_vel.rename(columns={'m': 'depth(m)', 'km/s': 'vp(km/s)', 'km/s.1': 'vs(km/s)'}, inplace=True)
U1519_vel[U1519_vel==(-999.25)]=np.nan
U1519_vel = U1519_vel.sort_values('depth(m)')

U1519_den = pd.read_csv('/Users/mckenziecarlson/pylith/pylith-4.1.1-macOS-10.15-x86_64/aotearoa/U1519_density.csv',skiprows=5)
U1519_den.rename(columns={'m': 'depth(m)', 'g/cm3': 'density(g/cm3)'}, inplace=True)
U1519_den[U1519_den==(-999.25)]=np.nan
U1519_den = U1519_den.sort_values('depth(m)')

## Define function for fitting piecewise data. It returns inflection points of piecewise function, can fit a line in between
## taken from https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a

def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)

## ------ Fit line to data using either degree-1 polynomial fits or piecewise linear ------- ##

# U1518 vs -----------------
idx = np.isfinite(U1518_vel["depth(m)"]) & np.isfinite(U1518_vel["vs(km/s)"])

x = U1518_vel["depth(m)"][idx]
y = U1518_vel["vs(km/s)"][idx]

c1 = np.polyfit(x, y, 1) # Calculate the polynomial fit (linear)
D1 = np.arange(0, 625, 0.2) # generate some grain diameter values
w1 = np.polyval(c1, D1)

# generate points in polynomial function
U1518_vs = w1
U1518_vs_depth = D1


# U1518 vp -------------------

idx = np.isfinite(U1518_vel["depth(m)"]) & np.isfinite(U1518_vel["vp(km/s)"])
x = U1518_vel["depth(m)"][idx]
y = U1518_vel["vp(km/s)"][idx]

# fit piecewise
px, py = segments_fit(x, y, 5)

x_range = np.arange(0, 625, 0.2)
y_range = np.interp(x_range, px, py)
plt.plot(y_range,x_range)

# generate points in piecewise function
U1518_vp = y_range
U1518_vp_depth = x_range


# U1518 density -----------------

idx = np.isfinite(U1518_vel["depth(m)"]) & np.isfinite(U1518_den["density(g/cm3)"])
x = U1518_den["depth(m)"][idx]
y = U1518_den["density(g/cm3)"][idx]

px, py = segments_fit(x, y, 5)

x_range = np.arange(0, 625, 0.2)
y_range = np.interp(x_range, px, py)

plt.plot(y_range,x_range)

# generate points in piecewise function
U1518_density = y_range
U1518_density_depth = x_range


# U1519 vs -----------------

idx = np.isfinite(U1519_vel["depth(m)"]) & np.isfinite(U1519_vel["vs(km/s)"])

x = U1519_vel["depth(m)"][idx]
y = U1519_vel["vs(km/s)"][idx]

c1 = np.polyfit(x, y, 1) # Calculate the polynomial fit (linear)
D1 = np.arange(0, 625, 0.2) # generate some grain diameter values
w1 = np.polyval(c1, D1)

# generate points in polynomial function
U1519_vs = w1
U1519_vs_depth = D1


# U1519 vp -----------------

idx = np.isfinite(U1519_vel["depth(m)"]) & np.isfinite(U1519_vel["vp(km/s)"])

x = U1519_vel["depth(m)"][idx]
y = U1519_vel["vp(km/s)"][idx]


c1 = np.polyfit(x, y, 1) # Calculate the polynomial fit (linear)
D1 = np.arange(0, 625, 0.2) # generate some grain diameter values
w1 = np.polyval(c1, D1)

# generate points in polynomial function
U1519_vp = w1
U1519_vp_depth = D1


# U1519 density -----------------

idx = np.isfinite(U1519_vel["depth(m)"]) & np.isfinite(U1519_den["density(g/cm3)"])
x = U1519_den["depth(m)"][idx]
y = U1519_den["density(g/cm3)"][idx]

px, py = segments_fit(x, y, 7)

# generate points in piecewise function
x_range = np.arange(0, 625, 0.2)
y_range = np.interp(x_range, px, py)
plt.plot(y_range,x_range)

U1519_density = y_range
U1519_density_depth = x_range


## ---------- Generate data for U1518 ---------- ##
## Use linear polynomal for vs, piecewise linear for vp and density

d = {'depth': U1518_vs_depth, 'vs': U1518_vs, 'vp': U1518_vp, 'density': U1518_density}

U1518_data_df = pd.DataFrame(data=d)

## Append a few depth points to the beginning to pad mesh boundary

data = []
# always inserting new rows at the first position - last row will be always on top    
data.insert(0, {'depth': -100.0, 'vs': 0.496940, 'vp': 1.781069, 'density':1.876504})
U1518_data_df = pd.concat([pd.DataFrame(data), U1518_data_df], ignore_index=True)

ind = np.arange(1,51).tolist()
U1518_data_df = U1518_data_df.drop(index=ind)
U1518_data_df = U1518_data_df.reset_index(drop=True)

U1518_data_df

## ---------- Generate data for U1519 ---------- ##
## Use linear polynomal for vs and vp, piecewise linear for density

d = {'depth': U1519_vs_depth, 'vs': U1519_vs, 'vp': U1519_vp, 'density': U1519_density}

U1519_data_df = pd.DataFrame(data=d)

## Append a few depth points to the beginning to pad mesh boundary

data = []
# always inserting new rows at the first position   
data.insert(0, {'depth': -100.0, 'vs': 0.120449, 'vp': 1.658870, 'density':1.912369})
U1519_data_df = pd.concat([pd.DataFrame(data), U1519_data_df], ignore_index=True)

U1519_data_df = U1519_data_df.drop(index=ind)
U1519_data_df = U1519_data_df.reset_index(drop=True)

U1519_data_df

## --------- Generat x,y data for points using topography and depth --------- ##

# Find index nearest boreholes and extend by 5 data points (values are depths of points nearest borehole seafloor)
end_idx = np.where(topo == -2.68720920e+03)[0][0] + 3
start_idx = np.where(topo == -1.00629570e+03)[0][0] - 3

x_range = np.linspace(topo[start_idx,0], topo[end_idx,0], 50)

# Interpolate y values so that vector is the same length as x values
y_vals_for_x_range = topo[start_idx:end_idx+1,:]
interpolator = interp1d(y_vals_for_x_range[:,0], y_vals_for_x_range[:,1], kind='linear')  # Linear interpolation

topo_range = interpolator(x_range)

## -------- Generate dataframe with vs, vp, and density at each point -------- ##

d = []
for i in range(0,len(U1518_data_df)):
    for j in range(0,len(x_range)-1):
        x = x_range[j]
        depth = U1518_data_df['depth'][i]
        y = topo_range[j] - depth

        # Linearly interpolate vs, vp, and density between two boreholes
        vs_range = np.linspace(U1519_data_df['vs'][i],U1518_data_df['vs'][i],len(x_range))
        vs = vs_range[j] * 1000

        vp_range = np.linspace(U1519_data_df['vp'][i],U1518_data_df['vp'][i],len(x_range))
        vp = vp_range[j] * 1000

        den_range = np.linspace(U1519_data_df['density'][i],U1518_data_df['density'][i],len(x_range))
        den = den_range[j] * 1000

        d.append(
            {
                'x_coord': x,
                'depth_coord': y,
                'vs': vs,
                'vp': vp,
                'density': den
            }
        )

data = pd.DataFrame(d)

## --------- Subsample data at every 20m depth ------------- ##
## depth interval is ~0.2m, so 0.2x100=20m (approx)

# Parameters
sample_size = 199
skip_size = sample_size * 100

# Create an empty list to hold the sampled data
sampled_data = []

# Iterate over the DataFrame in chunks
for start in range(0, len(data), sample_size + skip_size):
    end = start + sample_size
    sampled_data.append(data.iloc[start:end])

# Concatenate all the sampled chunks into a single DataFrame
sampled_df = pd.concat(sampled_data, ignore_index=True)

## -------- Save as csv ------- ##
sampled_df.to_csv('downsampled_20m_lwd.csv', index=False)