echo off

mpiexec -n 8 TWI -slp %1 -sca %2 -twi %3
