'''
Test if Tempest regridding gives the same result as GMAO tilefiles
The data regridded by GMAO tilefiles are generated outside (by MATLAB package),
and already exist in the sample_files directory.

J.W.Zhuang 2016/12
'''

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import tempest_remap

# create c2l regridding object from Tempest file
infile = "../offline_MAP/c48-to-lon72_lat46_MAP_DEPE_GMAO.nc"
c2l = tempest_remap.C2L(infile,72,46,48)

# create l2c regridding object from Tempest file
infile = "../offline_MAP/lon72_lat46-to-c48_MAP_DEPE_GMAO.nc"
l2c = tempest_remap.L2C(infile,72,46,48)

# read test data from file
datafile = 'sample_files/Projection_Test.nc'
fh =  Dataset(datafile, "r", format="NETCDF4")
LLdata = fh.variables['LLdata'][:] # this is the original lat-lon data
CSdata = fh.variables['CSdata_GMAO'][:] # regridded to CS by GMAO tilefile
CSdata = np.reshape(CSdata,(6,48,48)) # reshape 2D array to 3D
fh.close()

# apply regridding
# LLdata and LLdata_tmp should match, plus some error from regridding back and forth
LLdata_tmp = c2l.regrid(CSdata)
# CSdata and CSdata_tmp should be almost the same
CSdata_tmp = l2c.regrid(LLdata)


# plotting and comparing

plt.figure()
for n in range(6):
    plt.subplot(2,3,n+1)
    plt.pcolormesh(CSdata[n,:,:],vmax=6.0,vmin=0.0)
    plt.colorbar()
    plt.title('panel = '+str(n))
plt.suptitle('original CSdata')
plt.show(block=False)

plt.figure()
for n in range(6):
    plt.subplot(2,3,n+1)
    plt.pcolormesh(CSdata_tmp[n,:,:],vmax=6.0,vmin=0.0)
    plt.colorbar()
    plt.title('panel = '+str(n))
plt.suptitle('regridded CSdata')
plt.show(block=False)

plt.figure()
for n in range(6):
    plt.subplot(2,3,n+1)
    plt.pcolormesh(CSdata_tmp[n,:,:]-CSdata[n,:,:],vmax=0.05,vmin=-0.05, cmap='RdBu')
    plt.colorbar()
    plt.title('panel = '+str(n))
plt.suptitle('regridded - original')
plt.show(block=False)

plt.figure()
plt.pcolormesh(LLdata,vmax=6.0,vmin=0.0)
plt.colorbar()
plt.title('original LL data')
plt.show(block=False)

plt.figure()
plt.pcolormesh(LLdata_tmp,vmax=6.0,vmin=0.0)
plt.colorbar()
plt.title('regridded LL data')
plt.show(block=False)

plt.figure()
plt.pcolormesh(LLdata_tmp-LLdata,vmax=1.0,vmin=-1.0, cmap='RdBu')
plt.colorbar()
plt.title('regridded - original')
plt.show(block=True)#hold the x11 window


