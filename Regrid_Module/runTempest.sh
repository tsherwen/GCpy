#!/bin/bash

# Script for running Tempest. 
# Modified from /n/home08/elundgren/GCHP/tools/Tempest/runTempest.sh
# Jiawei Zhuang 2016/12

# --Set up your environment (using the GCHP ifort 13 bashrc works)--
# you can comment it out if it is already loaded, since loading takes time 
source GCHP.ifort13_openmpi_odyssey.bashrc 

# --input parameters--
# switches for regridding types. Can use both!
isC2L=true
isL2C=true
# grid resolution
nLon=72
nLat=46
nC=48
# additional
isDC=false # change from DE to DC?
isPC=false # change from PE to PC?
isGMAO=true # use 10 degree offset?

# --output directory--
outdest=$(pwd -P)/offline_MAP
# make output directory if it does not already exist
mkdir -p ${outdest}

# --tempest directory--
tempestdir='tempestremap'

# -------------------------------------
#   the followings are seldom changed
# -------------------------------------

# --create strings--

# grid type string and tempest additional options
if $isDC;then
    gridtype='DC'
    tempestopt=' --lonshift '
else
    gridtype='DE'
    tempestopt=' '
fi
if $isPC;then
    gridtype+='PC'
    tempestopt+=' --halfpole '
else
    gridtype+='PE'
fi
if $isGMAO;then
    gridtype+='_GMAO'
    tempestopt+=' --GMAOoffset '
else
    gridtype+='_NoOffset'
fi

# output file name
llStr=lon${nLon}_lat${nLat} 
out_ll=${llStr}_${gridtype}.g #grid type belongs to RLL mesh
out_cs=c${nC}.g
out_ov=${llStr}-and-c${nC}.g #overlap mesh. can be used for both C2L and L2C

# --Run Tempest--

cd $tempestdir/bin

./GenerateRLLMesh --lon ${nLon} --lat ${nLat} --file ${out_ll} ${tempestopt} #all options applied to RLL mesh
./GenerateCSMesh --res ${nC} --alt --file ${out_cs}
./GenerateOverlapMesh --a ${out_ll} --b ${out_cs} --out ${out_ov}

if $isC2L;then
    out_c2l=c${nC}-to-${llStr}_MAP_${gridtype}.nc
    ./GenerateOfflineMap --in_mesh ${out_cs} --out_mesh ${out_ll} --ov_mesh ${out_ov} --in_np 1 --out_map ${out_c2l}
    mv ${out_c2l} ${outdest}
fi

if $isL2L;then
    out_l2c=${llStr}-to-c${nC}_MAP_${gridtype}.nc
    ./GenerateOfflineMap --in_mesh ${out_ll} --out_mesh ${out_cs} --ov_mesh ${out_ov} --in_np 1 --out_map ${out_l2c}
    mv ${out_l2c} ${outdest}
fi

rm *.g # remove intermediate files for clarity

echo 'using addtional option:' ${tempestopt}
