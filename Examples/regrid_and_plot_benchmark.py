import os
import numpy as np
from netCDF4 import Dataset
from Benchmark_Module import Tracer_LLalready as TLL
from Benchmark_Module import File_Regrid
from Benchmark_Module.tracerlist import tracerlist_127

run_name='1mo_mean'
fname_gchp = 'GCHP.center.201307_monmean.nc4'
dname_gchp = 'gchp_c24_standard'
fname_gcc = 'GEOSCHEM_Diagnostics_Hrly.201308010000.nc'
dname_gcc = 'geosfp_4x5_standard'
work_dir = os.getcwd()
map_dir = '/n/regal/jacob_lab/elundgren/GC/std_1mo/v11-02b/util/offline_MAP'
plots_dir =os.path.join(work_dir,'plots')
outNlon = 288
outNlat = 181

def regrid_gchp_cs_to_ll(infile):
    outfile = infile.replace('.nc4','')+'.1x1.25.nc4'
    Mapfile = os.path.join(map_dir,'c24-to-288x181_DEPC_GMAO.nc')
    Nx = 24
    if os.path.exists(outfile):
        print('\nRegridded GCHP file already exists:',outfile)
    elif not os.path.exists(infile):
        print('\nMissing GCHP output file!',infile)
    else:
        print('\nRegridding GCHP file:',infile)
        File_Regrid.File_C2L(Mapfile,infile,Nx,outfile,outNlon,outNlat)
    return outfile

def regrid_gcc(infile):
    outfile = infile.replace('.nc4','')+'.1x1.25.nc'
    Mapfile = os.path.join(map_dir,'72x46_DEPC-to-288x181_DEPC.nc')
    inNlon = 72
    inNlat = 46
    if os.path.exists(outfile):
        print('\nRegridded GCC file already exists:',outfile)
    elif not os.path.exists(infile):
        print('\nMissing GCC output file!',infile)
    else:
        print('\nRegridding GCC file:',infile)
        File_Regrid.File_L2L(Mapfile,infile,inNlon,inNlat,outfile,outNlon,outNlat)
    return outfile

def compare_output(outdir,testname,file1,file2,tracerlist):
    bm = TLL.benchmark(outputdir=outdir+'/',shortname=testname)
    if os.path.basename(file1).startswith('GCHP'):
        bm.getdata1(file1,tag='GCHP',flip=False,tracerlist=tracerlist_127)
        print('file1 is GCHP')
    else:
        bm.getdata1(file1,tag='GCC',tracerlist=tracerlist_127)
        print('file1 is GCC')
    if os.path.basename(file2).startswith('GCHP'):
        bm.getdata2(file2,tag='GCHP',flip=False)
        print('file2 is GCHP')
    else:
        bm.getdata2(file2,tag='GCC')
        print('file2 is GCC')
    print('Creating comparison plots for',testname)
    bm.plot_all(plot_change=False,
                plot_surf=True,
                plot_500hpa=True,
                plot_zonal=True,
                plot_fracDiff=True)

if __name__ == "__main__":
    gchpfile = os.path.join(work_dir,dname_gchp,'OutputDir',fname_gchp)
    gchp_regridded = regrid_gchp_cs_to_ll(gchpfile)
    gcc_regridded = regrid_gcc(os.path.join(work_dir,dname_gcc,fname_gcc))

    # Check if files and plot directory exist
    if not os.path.isfile(gchp_regridded):
        print('WARNING!!!! Missing GCHP file',gchp_regridded)
    if not os.path.isfile(gcc_regridded):
        print('WARNING!!! Missing GCC file',gcc_regridded)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Generate plots (function defined above)
    compare_output(plots_dir,run_name,gchp_regridded,gcc_regridded,tracerlist=None)
