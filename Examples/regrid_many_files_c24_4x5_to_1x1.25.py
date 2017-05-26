# regrid_bench_c24_4x5_to_1x1.25.py
# Regrid GCHP cs c24 output and GCC 4x5 output to 1x1.25 netcdf file
# given directories for exclusion tests, isolation tests, and standard 1mo runs.
# I created this script for the preliminary GCHP v1.0.0 benchmark.
# Lizzie Lundgren, 3/31/17

import numpy as np
import os

from netCDF4 import Dataset
from Benchmark_Module import File_Regrid

# ODYSSEY NOTE: set env before starting:
#    module load python/3.4.1-fasrc01
#    source activate GCHP (virtual env your previously set up)
#    export PYTHONPATH=$PYTHONPATH:/n/regal/jacob_lab/elundgren/GCHP/benchmarks/v11-01/plot_tools/GCHPy

outNlon = 288
outNlat = 181
benchdir ='/n/regal/jacob_lab/elundgren/GCHP/benchmarks/v11-01' 
Mapdir = os.path.join(benchdir,'plot_tools/GCHPy/Regrid_Module/offline_MAP')

def regrid_gchp(datadir):
    if 'standard_1mo' in datadir:
        infilename = 'GCHP.center.20130801.nc4'
        outfilename ='GCHP.center.20130801.1x1.25.nc4'
    else:
        infilename = 'GCHP.center.20130702.nc4'
        outfilename = 'GCHP.center.20130702.1x1.25.nc4'
    outdir = os.path.join(datadir,'OutputDir')
    infile = os.path.join(outdir,infilename)
    outfile = os.path.join(outdir,outfilename)
    Mapfile = os.path.join(Mapdir,'c24-to-288x181_DEPC_GMAO.nc')
    Nx = 24
    if os.path.exists(outfile):
        print('\nRegridded GCHP file already exists:',outfile)
    elif not os.path.exists(infile):
        print('\nMissing GCHP output file!',infile)
    else:
        print('\nRegridding GCHP file:',infile)
        File_Regrid.File_C2L(Mapfile,infile,Nx,outfile,outNlon,outNlat)

def regrid_geosfp(datadir):
    if 'standard_1mo' in datadir:
        infilename = 'GEOSCHEM_Diagnostics_Hrly.201308010000.nc'
        outfilename = 'GEOSCHEM_Diagnostics_Hrly.201308010000.1x1.25.nc4'
    else:
        infilename = 'GEOSCHEM_Diagnostics_Hrly.201307020000.nc'
        outfilename = 'GEOSCHEM_Diagnostics_Hrly.201307020000.1x1.25.nc4'
    infile = os.path.join(datadir,infilename)
    outfile = os.path.join(datadir,outfilename) 
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

if __name__ == "__main__":
    maindir = os.path.join(benchdir,'bench_tests')
    testdirs = ['isolation_1day','exclusion_1day','standard_1mo']
    for testdir in testdirs:
        tests = os.listdir(os.path.join(maindir,testdir))
        for test in tests:
            datadir = os.path.join(maindir,testdir,test)
            if test.startswith('geosfp'):
                regrid_geosfp(datadir)
            elif test.startswith('gchp'):
                regrid_gchp(datadir)
            else: 
                print('unknown directory prefix: must be gchp or geosfp')
