import os
import numpy as np
from netCDF4 import Dataset
from Benchmark_Module import Tracer_LLalready as TLL
from Benchmark_Module import File_Regrid
from Benchmark_Module.tracerlist import tracerlist_127



def TMS_compare():
    """ """
    # --- Setup local variables
    work_dir = os.getcwd()
#    plots_dir =os.path.join(work_dir,'plots')
    plots_dir = work_dir
    outdir = work_dir
    data_dir = '/Users/tomassherwen/Google_Drive/Data/MUTD_iGEOS-Chem_output/'

    # import tracer lists
    from Benchmark_Module.tracerlist import tracerlist_150_v11_1g_HAL
    from Benchmark_Module.tracerlist import tracerlist_100_v10_HAL
    from Benchmark_Module.tracerlist import tracerlist_125
    from Benchmark_Module.tracerlist import tracerlist_103_v10_HAL

    # --- old test files from v10-01 
#    file1=data_dir+'iGEOSChem_3.0_v10/run/ctm.nc'
#    file2 = data_dir+'/iGEOSChem_3.0_v10/run.ClBrI.R.t22/ctm.nc'
    # get 1st data 
#    bm.getdata1(file1,tag='GCC',tracerlist=tracerlist_100)
    # get 2nd data
#    bm.getdata2(file2,tag='GCC')
    # --- v11-1g vs. WAH
#     run_name='v11_1g_vs_v11_1g_WAH'
#     testname = 'v11_1g_vs_v11_1g_WAH'
#     # Setup class
#     bm = TLL.benchmark(outputdir=outdir+'/',shortname=testname)
#     # directories for data 
#     file1 = data_dir + '/run.v11-1g.standard.FP.2014/'
#     file2 =  data_dir + '/iGEOSChem_5.0/run.WAH.0.1.0.1/'
#     file1, file2 = [ i+'ctm.nc' for i in [file1, file2] ]
#     # get 1st data 
#     bm.getdata1(file1,tag='v11-1g',tracerlist=tracerlist_125, avg_time=True)
#     # get 2nd data
#     bm.getdata2(file2,tag='WAH', avg_time=True)
    # --- "Cl+Br+I"(v10-01) vs. "WAH"(v11-1g)
#     run_name= 'v10_ClBrI_vs_v11_1g_WAH'
#     testname = 'v10_ClBrI_vs_v11_1g_WAH'
#     # Setup class
#     bm = TLL.benchmark(outputdir=outdir+'/',shortname=testname)
#     # directories for data 
#     file1 = data_dir +'iGEOSChem_4.0/run.XS.UPa.FP.EU.BC.II.FP.2014/'
#     file2 =  data_dir + '/iGEOSChem_5.0/run.WAH.0.1.0.1/'
#     file1, file2 = [ i+'ctm.nc' for i in [file1, file2] ]
#     # get 1st data     
#     bm.getdata1(file1,tag="Cl+Br+I",tracerlist=tracerlist_100_v10_HAL, \
#         avg_time=True)
#     # get 2nd data
#     bm.getdata2(file2,tag="WAH", avg_time=True)

    # --- "ClBrI"(2014) vs. "NIT"
    run_name= 'v10_ClBrI_vs_v10_ClBrI_NITS'
    testname = 'v10_ClBrI_vs_v10_ClBrI_NITS'
    # Setup class
    bm = TLL.benchmark(outputdir=outdir+'/',shortname=testname)
    # directories for data 
    file1 = data_dir +'iGEOSChem_4.0/run.XS.UPa.FP.EU.BC.II.FP.2014/'
    file2 =  data_dir + 'iGEOSChem_4.0/NITS_1year_output/'
    file1, file2 = [ i+'ctm.nc' for i in [file1, file2] ]
    # get 1st data     
    bm.getdata1(file1,tag="ClBrI(2014)",tracerlist=tracerlist_103_v10_HAL, \
        avg_time=True)
    # get 2nd data
    bm.getdata2(file2,tag="NITS", avg_time=True)

        
    # plot
    print('Creating comparison plots for',testname)
#    bm.plot_all(plot_surf=True,
#        plot_500hpa=False,
#        plot_zonal=False,
#        plot_fracDiff=False,
#        plot_change=False)
#     all plots
#     bm.plot_all(plot_change=False,
#                 plot_surf=True,
#                 plot_500hpa=True,
#                 plot_zonal=True,
#                 plot_zonal=True,
#                 plot_fracDiff=True)
    # without zonal
    bm.plot_all(plot_change=False,
                plot_surf=True,
                plot_500hpa=True,
                plot_zonal=False,
                plot_fracDiff=True)

    
def default_compare():
    """ Main driver setup by """
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

if __name__ == "__main__":
    TMS_compare()