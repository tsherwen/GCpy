# C2L or L2L conservative regridding
# for converting GCC and GCHP data to the same lat-lon mesh (typically finer). 

import numpy as np
from netCDF4 import Dataset

from Regrid_Module.python_interface import tempest_remap

def write_basic_info(fhout,Ntime,Nlev,Nlat,Nlon):

    timeid = fhout.createDimension("time",Ntime)
    levid = fhout.createDimension("lev",Nlev)
    latid = fhout.createDimension("lat",Nlat)
    lonid = fhout.createDimension("lon",Nlon)

    timearr = fhout.createVariable("time","f4",("time",))
    lonarr = fhout.createVariable("lon","f4",("lon",))
    latarr = fhout.createVariable("lat","f4",("lat",))
    levarr = fhout.createVariable("lev","f4",("lev",))

    timearr[:] = np.arange(1,Ntime+1)
    lonarr[:] = np.linspace(-180,180,Nlon,endpoint=False)
    latarr[:] = np.linspace(-90,90,Nlat,endpoint=True)
    levarr[:] = np.arange(1,Nlev+1)

def File_C2L(Mapfile,infile,Nx,outfile,Nlon_out,Nlat_out,flip=True):

    # initialize the weight once, and use it throughout the whole file
    c2l = tempest_remap.C2L(Mapfile,Nlon_out,Nlat_out,Nx)

    # open input data
    fhin = Dataset(infile, "r", format="NETCDF4")
    if Nx != len(fhin['lon'][:]) or Nx*6 != len(fhin['lat'][:]):
        # GCHP native output currently uses 'lon' for Nx and 'lat' for Ny=6*Nx
        # had better be changed in the future
        raise ValueError('dimension mismatch')
    Nlev = len(fhin['lev'][:])
    Ntime = len(fhin['time'][:])
    var = fhin.variables

    # create output file
    fhout = Dataset(outfile, "w", format="NETCDF4")
    write_basic_info(fhout,Ntime,Nlev,Nlat_out,Nlon_out) 

    # regrid the tracer field and write to file
    prefix = 'SPC_'
    for k,v in var.items():
        # skip the dimension variables
        if prefix in k:

            print('regridding',k)

            outdata = np.zeros([Ntime,Nlev,Nlat_out,Nlon_out])
            for itime in range(Ntime):
                outdata[itime,:,:,:] = c2l.regrid(v[itime,:,:,:])

            if flip: outdata=outdata[itime,::-1,:,:]

            outdata_to_file = fhout.createVariable(k,"f4",("time","lev","lat","lon"))
            outdata_to_file[:] = outdata 

    fhin.close()
    fhout.close()

def File_L2L(Mapfile,infile,Nlon_in,Nlat_in,outfile,Nlon_out,Nlat_out):
    pass

    l2l = tempest_remap.L2L(Mapfile,Nlon_in,Nlat_in,Nlon_out,Nlat_out)

    # open input data
    fhin = Dataset(infile, "r", format="NETCDF4")
    if Nlon_in != len(fhin['lon'][:]) or Nlat_in != len(fhin['lat'][:]):
        raise ValueError('dimension mismatch')
    Nlev = len(fhin['lev'][:])
    Ntime = len(fhin['time'][:])
    var = fhin.variables

    # create output file
    fhout = Dataset(outfile, "w", format="NETCDF4")
    write_basic_info(fhout,Ntime,Nlev,Nlat_out,Nlon_out) 

    # regrid the tracer field and write to file
    prefix = 'SPC_'
    for k,v in var.items():
        # skip the dimension variables
        if prefix in k:

            print('regridding',k)

            outdata = np.zeros([Ntime,Nlev,Nlat_out,Nlon_out])
            for itime in range(Ntime):
                outdata[itime,:,:,:] = l2l.regrid(v[itime,:,:,:])

            outdata_to_file = fhout.createVariable(k,"f4",("time","lev","lat","lon"))
            outdata_to_file[:] = outdata

    fhin.close()
    fhout.close()
