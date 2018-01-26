echo off

mpiexec -n 8 C:\Conda\Lib\site-packages\pydem\taudem\taudem_Windows\TauDEM\TauDEM5Exe\AreaDinf -ang %1 -sca %2
