echo off
path = C:\Program Files\Microsoft MPI\Bin;C:\Program Files\TauDEM\TauDEM5Exe;C:\Program Files\GDAL
mpiexec -n %1 DinfFlowDir.exe -ang %2 -slp %3 -fel %4