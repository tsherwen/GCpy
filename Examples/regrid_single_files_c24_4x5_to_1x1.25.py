# regrid_bench_c24_4x5_to_1x1.25.py
# Generic script to regrid GCHP c24 and GCC 4x5 to 1x1.25. 
# Lizzie Lundgren, 5/26/17

import numpy as np
import os
from netCDF4 import Dataset
from Benchmark_Module import File_Regrid

outNlon = 288
outNlat = 181

def regrid_gchp_cs_to_ll(datadir,infile):
    outfile = infile.replace('.nc4','')+'.1x1.25.nc4'
    outdir = os.path.join(datadir,'OutputDir')
    Mapfile = os.path.join(Mapdir,'c24-to-288x181_DEPC_GMAO.nc')
    Nx = 24
    if os.path.exists(outfile):
        print('\nRegridded GCHP file already exists:',outfile)
    elif not os.path.exists(infile):
        print('\nMissing GCHP output file!',infile)
    else:
        print('\nRegridding GCHP file:',infile)
        File_Regrid.File_C2L(Mapfile,infile,Nx,outfile,outNlon,outNlat)

def regrid_gcc(datadir,infile):
    outfile = infile.replace('.nc4','')+'.1x1.25.nc4'
    Mapfile = os.path.join(Mapdir,'72x46_DEPC-to-288x181_DEPC.nc')
    inNlon = 72
    inNlat = 46
    if os.path.exists(outfile):
        print('\nRegridded GCC file already exists:',outfile)
    elif not os.path.exists(infile):
        print('\nMissing GCC output file!',infile)
    else:
        print('\nRegridding GCC file:',infile)
        File_Regrid.File_L2L(Mapfile,infile,inNlon,inNlat,outfile,outNlon,outNlat)

# main gets executed if you do 'python ./regrid_4x5_to_1x1.25.py'
if __name__ == "__main__":

    # Configurables
    workdir ='/n/regal/jacob_lab/elundgren/GC/testruns/v11-02b'
    Mapdir = os.path.join(workdir,'offline_MAP') # tempest output required
    gchpdirname = 'gchp_allOff_1hr'
    gccdirname = 'gcc_allOff_1hr'
    gchpfilename = 'GCHP.center.20130701.nc4'
    gccfilename = 'GEOSCHEM_Diagnostics_Hrly.201307010100.nc4'

    gcc_file = os.path.join(workdir,gccdirname,gccfilename)
    gchp_file = os.path.join(workdir,gchpdirname,gchpfilename)
    regrid_gcc(gcc_dir)
    regrid_gchp_cs_to_ll(gchp_dir)
