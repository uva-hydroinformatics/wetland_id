# -*- coding: utf-8 -*-
"""
Modules perform Fill, Impact Reduction Appraoch (IRA), and A* lest-cost hydro conditioning,
and calculate Laplacian curvature, topographic wetness index (TWI), and cartographic depth-to-water (DTW),
as well as the necessary inputs for these wetness indices.  

Created on Thu Mar 29 10:50:01 2018
@author: Gina O'Neil
"""

import subprocess
import os
from osgeo import gdal, gdal_array
import numpy as np
import scipy
from scipy import stats
import wetland_id_defaults as default
import sys
from scipy import ndimage
from raster_array_funcs import *
import grass.script as g
import grass.script.setup as gsetup
import shutil
import wetland_tool_grass as grass
import time
import multiprocessing


def calc_slope_curv(dem_arr, pix_res, curvatureCalcMethod = 'laplacian'):
    #pygeonet code adapted
    
    start_t = time.time()
    print "Beginning slope and curvature calculations... \n"
    
    ##must pass a masked or non-cleaned dem array
    slopeXArray,slopeYArray = np.gradient(dem_arr, pix_res)
    slope_arr = np.sqrt(slopeXArray**2 + slopeYArray**2)  
    
    #only performed to calculate statistics
    slopeMagnitudeDemArrayQ = slope_arr

    slopeMagnitudeDemArrayQ.flatten()

    slopeMagnitudeDemArrayQ = slopeMagnitudeDemArrayQ[~np.isnan(slopeMagnitudeDemArrayQ)]

    # Computation of statistics of slope
    print ' slope statistics'
    print ' angle min (degrees):', np.arctan(np.percentile(slopeMagnitudeDemArrayQ,0.1))*180/np.pi
    print ' angle max (degrees):', np.arctan(np.percentile(slopeMagnitudeDemArrayQ,99.9))*180/np.pi
    print ' mean slope (rise/run):', np.nanmean(slope_arr[:])
    print ' stdev slope (rise/run):', np.nanstd(slope_arr[:]), '\n'
    
    #add small constant to avoid no data values in subsequent processes
    slope_arr_add = slope_arr + default.slope_const
    
    if curvatureCalcMethod == 'geometric':
        #Geometric curvature
        print ' using geometric curvature \n'
        gradXArrayT = np.divide(slopeXArray,slope_arr_add)
        gradYArrayT = np.divide(slopeYArray,slope_arr_add)
    elif curvatureCalcMethod=='laplacian':
        # do nothing..
        print ' using laplacian curvature'
        gradXArrayT = slopeXArray
        gradYArrayT = slopeYArray
    
    gradGradXArray,tmpy = np.gradient(gradXArrayT,pix_res)
    tmpx,gradGradYArray = np.gradient(gradYArrayT,pix_res)
    curv_arr = gradGradXArray + gradGradYArray
    curv_arr[np.isnan(curv_arr)] = -9999
    del tmpy, tmpx
    
    end_t = time.time()
    print "Calculations complete, execution time: %.2f \n" %(end_t - start_t)
    
    return slope_arr_add, curv_arr

#curvature function adapted from pygeonet module: compute_dem_curvature
def calc_curvature(dem_arr, out_dir, dem_meta, curvatureCalcMethod = 'laplacian'):
    pix_res = float(dem_meta['pix_res'])
    gradXArray, gradYArray = np.gradient(dem_arr, pix_res)
    slopeArrayT = np.sqrt(gradXArray**2 + gradYArray**2)
    if curvatureCalcMethod == 'geometric':
        #Geometric curvature
        print ' using geometric curvature'
        gradXArrayT = np.divide(gradXArray,slopeArrayT)
        gradYArrayT = np.divide(gradYArray,slopeArrayT)
    elif curvatureCalcMethod=='laplacian':
        # do nothing..
        print ' using laplacian curvature'
        gradXArrayT = gradXArray
        gradYArrayT = gradYArray
    
    gradGradXArray,tmpy = np.gradient(gradXArrayT,pix_res)
    tmpx,gradGradYArray = np.gradient(gradYArrayT,pix_res)
    curvatureDemArray = gradGradXArray + gradGradYArray
    curvatureDemArray[np.isnan(curvatureDemArray)] = 0
    del tmpy, tmpx
    
    #write to geotiff
    curv_tif = array_to_geotif(curvatureDemArray, dem_meta, out_dir, (default.input_dem + default.cv_suf))
    
    return curv_tif

#tuadem  
#TODO figure out how to call commands with os.system...getting errors unless calling commands from within batch files
def calc_fill(dem_in, out_dir, n_proc=default.n_proc):
    # taudem pitremove to create depressionless dem
    print "Beginning TauDEM pitremove..."
    dem_base = dem_in.split('\\')[-1]
    fill_name = dem_base[:-4] + default.z_fill_suf
    #build command
    fill_out = os.path.join(out_dir, fill_name)
    if not os.path.exists(fill_out):
        batch_args = " %s %s %s" %(str(n_proc), dem_in, fill_out)
        cmd = default.taudem_fill + batch_args
        
        # Submit command to operating system
        os.system(cmd)
        # Capture the contents of shell command and print
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        message = "\n"
        for line in process.stdout.readlines():
            message = message + line
        print message
    return fill_out

##This is modified from geonet script but here we use grassgis r.hydrodem which implements Lind.&Creed 2005 IRA
def calc_at_twi(dem_in):
    print "Beginning GRASS GIS TWI calculations using A* method... \n"
    start_t = time.time()
    twi_at = grass.grass_At(dem_in)
    end_t = time.time()
    print "Calculations complete, execution time: %.2f \n" %(end_t - start_t)
    return twi_at

def calc_ira(dem_in):
    print "Beginning GRASS GIS hydro conditioning using Lindsey & Creed IRA method... \n"
    start_t = time.time()
    ira_out = grass.grass_IRA(dem_in)
    end_t = time.time()
    print "Calculations complete, execution time: %.2f \n" %(end_t - start_t)
    return ira_out

def calc_ira_twi(dem_in):
    print "Beginning GRASS GIS hydro conditioning using Lindsey & Creed IRA method... \n"
    start_t = time.time()
    dem_breach = grass.grass_breach(dem_in)
    print "Beginning GRASS GIS r.terraflow flow accumulation calculations... \n"
    ira_twi = grass.grass_ira_twi(dem_breach)
    end_t = time.time()
    print "Calculations complete, execution time: %.2f \n" %(end_t - start_t)
    return ira_twi
    
#needs to be passed a conditioned dem
def calc_fdr(dem_cond, out_dir, n_proc=default.n_proc):
    print "Beginning TauDEM D-infinity flow direction and slope calculations..."
    dem_cond_base = dem_cond.split('\\')[-1]
    slp_name = dem_cond_base[:-4] + default.slp_suf
    fdr_name = dem_cond_base[:-4] + default.fdr_suf
    
    #build command
    slp_out = os.path.join(out_dir, slp_name)
    fdr_out = os.path.join(out_dir, fdr_name)
    if not os.path.exists(fdr_out):
        batch_args = " %d %s %s %s" %(n_proc, fdr_out, slp_out, dem_cond)
        cmd = default.taudem_fdrslp + batch_args
        
        # Submit command to operating system
        os.system(cmd)
        # Capture the contents of shell command and print it to the arcgis dialog box
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        message = "\n"
        for line in process.stdout.readlines():
            message = message + line
        print message

    return slp_out, fdr_out

def calc_sca(fdr, n_proc=default.n_proc):    
    # taudem d-inf flow accumulation calculation
    #sca = specific catchment area = (flow accumulation * pix res)
    print "Beginning TauDEM specific catchment area..."
    
    #build command
    sca_out = fdr.replace(default.fdr_suf, default.sca_suf)
    
    if not os.path.exists(sca_out):
        batch_args = " %d %s %s" %(n_proc, fdr, sca_out)
        cmd = default.taudem_uca + batch_args
        
        # Submit command to operating system
        os.system(cmd)
        # Capture the contents of shell command and print it to the arcgis dialog box
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        message = "\n"
        for line in process.stdout.readlines():
            message = message + line
        print message
    return sca_out

def calc_twi(slp, sca, n_proc=default.n_proc):
    # taudem d-inf twi
    #should use slp raster with small constant added to avoid no data values in TWI
    print "Beginning TauDEM TWI calculation..."

    #check for zero values in slope
#    slp_arr, slp_meta = geotif_to_array(slp)
#    slp_name = slp.split("\\")[-1]
#    if 0 in slp_arr:        
#        #add small constant to avoid no data values in subsequent processes
#        slp_arr_add = slp_arr + default.slope_const
##        slp_arr_add_clean = np.ma.filled(clean_array(slp_arr_add), np.nan) #shortcut from clipping NaNs        
#        #write to geotiff
#        slp_add_tif = array_to_geotif(slp_arr_add, slp_meta, default.roi_vars, \
#                                   (slp_name[:-4] + "_slpnztd.tif"))
#    
#    else:
#        slp_add_tif = slp
    
    slp_add_tif = slp
    
    #build command
    twi_out = sca.replace(default.sca_suf, default.twi_suf) 
    if not os.path.exists(twi_out):
        batch_args = " %d %s %s %s" %(n_proc, slp_add_tif, sca, twi_out)
        cmd = default.taudem_TWI + batch_args
    
         # Submit command to operating system
        os.system(cmd)
        # Capture the contents of shell command and print it to the arcgis dialog box
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        message = "\n"
        for line in process.stdout.readlines():
            message = message + line
        print message    
    
    return twi_out

def calc_dtw(cost_tif, source_shp):
    dtw, dtw_dir = grass.grass_cost(cost_tif, source_shp)
    return dtw, dtw_dir

    

def comp_bands(file_dir, var_list, comp_name):
    """Given a list of variable geotiffs, merges variables into a multiband geotiff
    where each band contains data for individual variables """
    
    print "Beginning composite bands..."
    #need funky formatting to add double quotes around file paths to cmd    
    var_f = []
#    clip_bounds = default.roi_lim
    for v in range(len(var_list)):
#        var_list_clip = clip_geotif(var_list[v], default.roi_comps, clip_bounds)
#        var_f.append("\"%s\" " %(var_list_clip))
        var_f.append("\"%s\" " %(var_list[v]))
    comp_path_name = os.path.join(file_dir, comp_name)

    path_to_gdal_script = r"C:\Anaconda2\Lib\site-packages\osgeo\scripts\gdal_merge.py"

    cmd = "python %s -init \"-9999. -9999. -9999.\" -o \"%s\" " %(path_to_gdal_script, comp_path_name) \
    + "".join([x for x in var_f]) + "-separate -co COMPRESS=LZW -ot Float32" 
    
    print "Running gdal command: \n %s " %(cmd)
    subprocess.call(cmd, shell=True)
    return comp_path_name
    
