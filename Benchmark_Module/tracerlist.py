'''
A tracerlist with the same tracer order as in input.geos,
to force getdata1 and the subsequent plots to use this sequence,
no matter how the tracers in the netCDF file are ordered.

Example:
from Benchmark_Module import Tracer_LLalready as TLL
from Benchmark_Module.tracerlist import tracerlist_125
...
bm = TLL.benchmark(outputdir='plot/',shortname=shortname,longname=longname)
bm.getdata1(filename1,tracerlist=tracerlist_125,tag='GCHP')

Note:
1)
The netCDF file(filename1) should contain all the tracers in the tracerlist.
2)
The list is simply created by
$sed -n -e 26,150p input.geos > tracerlist.py

Then and fix the format by using the vim's virtual block (control+v)
and vim's command
:1,125s/$/',/g
3)
tracerlist_125 is compatible with GCHP v1.0.0 (v11-01 public).
tracerlist_127 is compatible with GCHP v1.1.0 (v11-02b).
'''

tracerlist_125 = [
        'NO',
        'O3',
        'PAN',
        'CO',
        'ALK4',
        'ISOP',
        'HNO3',
        'H2O2',
        'ACET',
        'MEK',
        'ALD2',
        'RCHO',
        'MVK',
        'MACR',
        'PMN',
        'PPN',
        'R4N2',
        'PRPE',
        'C3H8',
        'CH2O',
        'C2H6',
        'N2O5',
        'HNO4',
        'MP',
        'DMS',
        'SO2',
        'SO4',
        'SO4s',
        'MSA',
        'NH3',
        'NH4',
        'NIT',
        'NITs',
        'BCPI',
        'OCPI',
        'BCPO',
        'OCPO',
        'DST1',
        'DST2',
        'DST3',
        'DST4',
        'SALA',
        'SALC',
        'Br2',
        'Br',
        'BrO',
        'HOBr',
        'HBr',
        'BrNO2',
        'BrNO3',
        'CHBr3',
        'CH2Br2',
        'CH3Br',
        'MPN',
        'ISOPND',
        'ISOPNB',
        'MOBA',
        'PROPNN',
        'HAC',
        'GLYC',
        'MVKN',
        'MACRN',
        'RIP',
        'IEPOX',
        'MAP',
        'NO2',
        'NO3',
        'HNO2',
        'N2O',
        'OCS',
        'CH4',
        'BrCl',
        'HCl',
        'CCl4',
        'CH3Cl',
        'CH3CCl3',
        'CFC113',
        'CFC114',
        'CFC115',
        'HCFC123',
        'HCFC141b',
        'HCFC142b',
        'CFC11',
        'CFC12',
        'HCFC22',
        'H1211',
        'H1301',
        'H2402',
        'Cl',
        'ClO',
        'HOCl',
        'ClNO3',
        'ClNO2',
        'ClOO',
        'OClO',
        'Cl2',
        'Cl2O2',
        'H2O',
        'MTPA',
        'LIMO',
        'MTPO',
        'TSOG1',
        'TSOG2',
        'TSOG3',
        'TSOG0',
        'TSOA1',
        'TSOA2',
        'TSOA3',
        'TSOA0',
        'ISOG1',
        'ISOG2',
        'ISOG3',
        'ISOA1',
        'ISOA2',
        'ISOA3',
        'BENZ',
        'TOLU',
        'XYLE',
        'ASOG1',
        'ASOG2',
        'ASOG3',
        'ASOAN',
        'ASOA1',
        'ASOA2',
        'ASOA3'
        ]

tracerlist_127 = [
        'NO',
        'O3',
        'PAN',
        'CO',
        'ALK4',
        'ISOP',
        'HNO3',
        'H2O2',
        'ACET',
        'MEK',
        'ALD2',
        'RCHO',
        'MVK',
        'MACR',
        'PMN',
        'PPN',
        'R4N2',
        'PRPE',
        'C3H8',
        'CH2O',
        'C2H6',
        'N2O5',
        'HNO4',
        'MP',
        'DMS',
        'SO2',
        'SO4',
        'SO4s',
        'MSA',
        'NH3',
        'NH4',
        'NIT',
        'NITs',
        'BCPI',
        'OCPI',
        'BCPO',
        'OCPO',
        'DST1',
        'DST2',
        'DST3',
        'DST4',
        'SALA',
        'SALC',
        'Br2',
        'Br',
        'BrO',
        'HOBr',
        'HBr',
        'BrNO2',
        'BrNO3',
        'CHBr3',
        'CH2Br2',
        'CH3Br',
        'MPN',
        'ISOPND',
        'ISOPNB',
        'MOBA',
        'PROPNN',
        'HAC',
        'GLYC',
        'MVKN',
        'MACRN',
        'RIP',
        'IEPOX',
        'MAP',
        'NO2',
        'NO3',
        'HNO2',
        'N2O',
        'OCS',
        'CH4',
        'BrCl',
        'HCl',
        'CCl4',
        'CH3Cl',
        'CH3CCl3',
        'CFC113',
        'CFC114',
        'CFC115',
        'HCFC123',
        'HCFC141b',
        'HCFC142b',
        'CFC11',
        'CFC12',
        'HCFC22',
        'H1211',
        'H1301',
        'H2402',
        'Cl',
        'ClO',
        'HOCl',
        'ClNO3',
        'ClNO2',
        'ClOO',
        'OClO',
        'Cl2',
        'Cl2O2',
        'H2O',
        'MTPA',
        'LIMO',
        'MTPO',
        'TSOG1',
        'TSOG2',
        'TSOG3',
        'TSOG0',
        'TSOA1',
        'TSOA2',
        'TSOA3',
        'TSOA0',
        'ISOG1',
        'ISOG2',
        'ISOG3',
        'ISOA1',
        'ISOA2',
        'ISOA3',
        'BENZ',
        'TOLU',
        'XYLE',
        'ASOG1',
        'ASOG2',
        'ASOG3',
        'ASOAN',
        'ASOA1',
        'ASOA2',
        'ASOA3',
        'EOH',
        'MGLY'
        ]
                   
