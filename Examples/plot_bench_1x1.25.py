# plot_bench_1x1.25.py
# Create comparisons plots between GCHP and GCC regridded 1x1.25 outputs
# given directories contains 1-day exclusion tests, 1-day isolation tests, 
# and standard 1-month runs.
# I created this script for the preliminary GCHP v1.0.0 benchmark.
# Lizzie Lundgren, 3/31/17

import os
from Benchmark_Module import Tracer_LLalready as TLL

# ODYSSEY NOTE: make sure you set env before starting:
#    module load python/3.4.1-fasrc01
#    source activate GCHP (previously set up virtual env)
#    export PYTHONPATH=$PYTHONPATH:/n/regal/jacob_lab/elundgren/GCHP/benchmarks/v11-01/plot_tools/GCHPy

benchdir ='/n/regal/jacob_lab/elundgren/GCHP/benchmarks/v11-01'
plotsdir =os.path.join(benchdir,'plots')
gchpdirprefix = 'gchp_c24_standard_'
gccdirprefix = 'geosfp_4x5_standard_'

def bench_plot(outdir,testname,gchpfile,gccfile):
    bm = TLL.benchmark(outputdir=outdir+'/',
                       shortname=testname)
    bm.getdata1(gchpfile,tag='GCHP',flip=False)
    bm.getdata2(gccfile,tag='GCC')
    print('Creating benchmark plots for',testname)
    bm.plot_all(plot_change=False,
                plot_surf=True,
                plot_500hpa=True,
                plot_zonal=True,
                plot_fracDiff=True)

if __name__ == "__main__":
    maindir = os.path.join(benchdir,'bench_tests')
    testdirs = ['isolation_1day','exclusion_1day','standard_1mo']
    for testdir in testdirs:
        tests = os.listdir(os.path.join(maindir,testdir))
        for test in tests:
            if test.startswith('gchp'):

                # To plot for just one case, uncomment this and edit as needed
                #if 'advectionOff' not in test:
                #    continue

                testname = test[len(gchpdirprefix):]

                if 'standard_1mo' in test:
                    gchpfilename = 'GCHP.center.20130801.1x1.25.nc4'
                    gccfilename = 'GEOSCHEM_Diagnostics_Hrly.201308010000.1x1.25.nc4'
                else:
                    gchpfilename = 'GCHP.center.20130702.1x1.25.nc4'
                    gccfilename = 'GEOSCHEM_Diagnostics_Hrly.201307020000.1x1.25.nc4'

                gchpdir = os.path.join(maindir,testdir,test,'OutputDir')
                gccdir =  os.path.join(maindir,testdir,gccdirprefix+testname)

                gchpfile = os.path.join(gchpdir,gchpfilename)
                if not os.path.isfile(gchpfile):
                    print('WARNING!!!! Missing GCHP file for test',testname)
                    print ('Skipping to next test...')
                    continue

                gccfile = os.path.join(gccdir,gccfilename)
                if not os.path.isfile(gchpfile):
                    print('WARNING!!! Missing GCC file for test',testname)
                    print('Skipping to next test..')
                    continue

                testplotsdir = os.path.join(plotsdir,testdir,testname)
                if not os.path.exists(testplotsdir):
                    os.makedirs(testplotsdir)

                bench_plot(testplotsdir,testname,gchpfile,gccfile)

