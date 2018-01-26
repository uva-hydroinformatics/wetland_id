echo off
mpiexec -n 8 C:\Conda\Lib\site-packages\pydem\taudem\taudem_Windows\TauDEM\TauDEM5Exe\D8FlowDir -p %1 -sd8 %2 -fel %3
