# compare_two_runs.py
# Generic script to create comparison plots between lat/lon data.
# For GCHP, use regrid script to regrid from cs prior to running this.
# Lizzie Lundgren, 5/26/17

import os
from Benchmark_Module import Tracer_LLalready as TLL

# you can import this function to another script
def compare_output(outdir,testname,file1,file2,tracerlist):
    bm = TLL.benchmark(outputdir=outdir+'/',shortname=testname)
    if file1.startswith('GCHP'):
        bm.getdata1(gchpfile,tag='GCHP',flip=False,tracerlist=tracerlist)
    else:
        bm.getdata1(gccfile,tag='GCC',tracerlist=tracerlist)
    if file2.startswith('GCHP'):
        bm.getdata2(gchpfile,tag='GCHP',flip=False)
    else:
        bm.getdata2(gccfile,tag='GCC')
    print('Creating comparison plots for',testname)
    bm.plot_all(plot_change=False,
                plot_surf=True,
                plot_500hpa=True,
                plot_zonal=True,
                plot_fracDiff=True)

# main gets executed if you do 'python ./compare_two_runs.py'
if __name__ == "__main__":

    # Configurables
    workdir ='/path/to/your/working/dir'
    testname='test_scenario_description__no_spaces'
    plotsdir =os.path.join(workdir,'plots',testname)
    gchpdirname = 'gchp_dir_name'
    gccdirname = 'gcc_dir_name'
    gchpfilename = 'GCHP.center.20130701.1x1.25.nc4'
    gccfilename = 'GEOSCHEM_Diagnostics_Hrly.201307010100.1x1.25.nc4'
    tracerlist=['O3','NO'] # or use None for all species

    # Check if files and plot directory exist
    gchpfile = os.path.join(workdir,gchpdirname,'OutputDir',gchpfilename)
    if not os.path.isfile(gchpfile):
        print('WARNING!!!! Missing GCHP file',gchpfile)
    gccfile = os.path.join(workdir,gccdirname,gccfilename)
    if not os.path.isfile(gchpfile):
        print('WARNING!!! Missing GCC file',gccfile)
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)

    # Generate plots (function defined above)
    compare_output(plotsdir,testname,gchpfile,gccfile,tracerlist)

