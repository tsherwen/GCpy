# GCPy

A comprehensive python tool for the GEOS-Chem model with the focus on:

1. Benchmark suite for both GEOS-Chem classic and GCHP.

2. Visualization tools for data on either the lat/lon or cube-sphere grids.

3. Regridding tools for both lat/lon and cubed-sphere grids.


Originally developed by Jiawei Zhuang 03/25/2017

### Environment setup:

Example of first time setup on the Harvard Odyssey cluster. NOTE: can be adapted to your home system.


```
#!unix

$ module load python/3.4.1-fasrc01
$ conda create -n GCHP --clone="$PYTHON_HOME" (using name GCHP is just an example; you can name your virtual env whatever you like)
$ source activate GCHP
$ conda install Basemap
$ conda install netcdf4
$ export PYTHONPATH=$PYTHONPATH:{your_path_to_GCHPy}
```

On subsequent sessions:


```
#!unix

$ module load python/3.4.1-fasrc01
$ source activate GCHP
$ export PYTHONPATH=$PYTHONPATH:/your/path/to/GCHPy
```