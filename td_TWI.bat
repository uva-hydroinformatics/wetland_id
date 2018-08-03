echo off
path = C:\Program Files\Microsoft MPI\Bin;C:\Program Files\TauDEM\TauDEM5Exe;C:\Program Files\GDAL
mpiexec -n %1 TWI.exe -slp %2 -sca %3 -twi %4
