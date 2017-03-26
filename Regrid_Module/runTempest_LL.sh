#!/bin/bash

# For LatLon <-> LatLon regridding
# This simple task can be done by many other packages,
# but we use Tempest for everything for consistency.
# Also, using pre-generated regridding weights is much faster than
# doing all the stuff online.

# --Set up your environment (using the GCHP ifort 13 bashrc works)--
# you can comment it out if it is already loaded, since loading takes time 
source GCHP.ifort13_openmpi_odyssey.bashrc 

# --input parameters--
# Here we don't need the GMAO offset option as in the C&L regridding.
# Remember that the 10-degree offset is a CubeSphere issue, but we just 
# shift the lat-lon grid instead to get the same relative positioning.

# switches for regridding types. Can use both!
is1to2=true
is2to1=false

# input grid
nLon1=72
nLat1=46
isDC1=false # change from DE to DC?
isPC1=true # change from PE to PC?

# output grid
nLon2=288
nLat2=181
isDC2=false # change from DE to DC?
isPC2=true # change from PE to PC?

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
if $isDC1;then
    gridtype1='DC'
    tempestopt1=' --lonshift '
else
    gridtype1='DE'
    tempestopt1=' '
fi
if $isDC2;then
    gridtype2='DC'
    tempestopt2=' --lonshift '
else
    gridtype2='DE'
    tempestopt2=' '
fi
if $isPC1;then
    gridtype1+='PC'
    tempestopt1+=' --halfpole '
else
    gridtype1+='PE'
fi
if $isPC2;then
    gridtype2+='PC'
    tempestopt2+=' --halfpole '
else
    gridtype2+='PE'
fi

# output file name
llStr1=${nLon1}x${nLat1}_${gridtype1} 
llStr2=${nLon2}x${nLat2}_${gridtype2}
out_ll1=${llStr1}.g #grid type belongs to RLL mesh
out_ll2=${llStr2}.g #grid type belongs to RLL mesh
out_ov=${llStr1}-and-${llStr2}.g #overlap mesh. can be used for both direction 

# --Run Tempest--

cd $tempestdir/bin

./GenerateRLLMesh --lon ${nLon1} --lat ${nLat1} --file ${out_ll1} ${tempestopt1} #all options applied to RLL mesh
./GenerateRLLMesh --lon ${nLon2} --lat ${nLat2} --file ${out_ll2} ${tempestopt2} #all options applied to RLL mesh
./GenerateOverlapMesh --a ${out_ll1} --b ${out_ll2} --out ${out_ov}

if $is1to2;then
    out_1to2=${llStr1}-to-${llStr2}.nc
    ./GenerateOfflineMap --in_mesh ${out_ll1} --out_mesh ${out_ll2} --ov_mesh ${out_ov} --in_np 1 --out_map ${out_1to2}
    mv ${out_1to2} ${outdest}
fi

if $is2to1;then
    out_2to1=${llStr2}-to-${llStr1}.nc
    ./GenerateOfflineMap --in_mesh ${out_ll2} --out_mesh ${out_ll1} --ov_mesh ${out_ov} --in_np 1 --out_map ${out_2to1}
    mv ${out_2to1} ${outdest}
fi

rm *.g # remove intermediate files for clarity

echo 'using addtional option:' 
echo ${tempestopt1} '(for grid1)'
echo ${tempestopt2} '(for grid2)'
