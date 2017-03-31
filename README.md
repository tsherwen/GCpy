# GCHPy

A comprehensive python tool for the GEOS-Chem High-Performance (GCHP) with the focus on:

1) Cube-Sphere<->Lat-Lon conservative regridding without external grid description files. (Make use of the Tempest regridding package)

2) Direct visualization of data on the cube-sphere grid.

3) Benchmark suite for comparing GCHP and GEOSChem-classic.

### Environment setup:

On Odyssey, first time setup:

$ module load python/3.4.1-fasrc01</br>
$ conda create -n GCHP --clone="$PYTHON_HOME"

$ source activate GCHP

$ conda install Basemap

$ condal install netcdf4

$ export PYTHONPATH=$PYTHONPATH:{your_path_to_GCHPy}

On Odyssey, subsequent sessions:

$ module load python/3.4.1-fasrc01

$ source activate GCHP

$ export PYTHONPATH=$PYTHONPATH:{your_path_to_GCHPy}

Originally developed by Jiawei Zhuang 03/25/2017