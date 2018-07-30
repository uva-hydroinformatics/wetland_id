# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:39:59 2018

@author: Gina O'Neil
"""

from scipy import signal
from scipy import stats
from osgeo import gdal, gdal_array
import numpy as np
import os
import subprocess
from raster_array_funcs import *
from skimage import filters, util
from skimage import data, img_as_float
from scipy.stats.mstats import mquantiles
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import sys
import time
import pandas as pd
import wetland_id_defaults as default

def calc_filt_params(dem_meta, smoothing_width):
    pix_res = float(dem_meta['pix_res'])
    filt_window = int(default.smoothing_width / pix_res)  
    return filt_window

def med_filt(dem_arr, window):
    print "Beginning median filtering with window size: %d..." %(window)
    start_t = time.time()
    dem_med_arr = ndimage.median_filter(dem_arr, size = window, mode = 'reflect') #ndimage is MUCH faster than scipiy.signal
    end_t = time.time()
    print "Median filtering complete, execution time: %.2f \n" %(end_t - start_t)
    return dem_med_arr

def mean_filt(dem_arr, window):
    print "Beginning mean filtering with window size: %d..." %(window)
    start_t = time.time()
    dem_mean_arr = ndimage.uniform_filter(dem_arr, size=window, mode='reflect')
    end_t = time.time()
    print "Mean filtering complete, execution time: %.2f \n" %(end_t - start_t)
    return dem_mean_arr


def gaus_filt(dem_arr): #gaus needs nan values for boundaries
    print "Beginning gaussian filtering with standard deviation: %.4f..." %(default.gaus_stdev)
    start_t = time.time()
    dem_gaus_arr = ndimage.gaussian_filter(dem_arr, sigma = float(default.gaus_stdev), mode = 'reflect')
    end_t = time.time()
    print "Gaussian filtering complete, execution time: %.2f \n" %(end_t - start_t)
    return dem_gaus_arr


def pm_filt_slp(dem_arr, dem_meta, niter = default.pm_n_iter):
   
    """ !!! perona malik code is executed using pygeonet code: anisodiff and lambda_nonlinear_filter!!! 
    source: Sangireddy et al., 2016 http://dx.doi.org/10.1016/j.envsoft.2016.04.026"""
    
    #passing masked dem and nan dem yield same results, but masked dem takes 4X as long...extra step of filling clean masked dem with NaNs
    
    print "Beginning perona malik filtering with %d iterations..." %(niter)
    start_t = time.time()
    pixel_res = float(dem_meta['pix_res'])
    dem_in = np.ma.filled(dem_arr, np.nan)
    
    #perform pygeonet lambda_nonlinear_filter
    slopeXArray, slopeYArray = np.gradient(dem_in, pixel_res)
    slopeMagnitudeDemArray = np.sqrt(slopeXArray**2 + slopeYArray**2)
    print 'DEM slope array shape:', slopeMagnitudeDemArray.shape
    
    # Computation of the threshold lambda used in Perona-Malik nonlinear
    # filtering. The value of lambda (=edgeThresholdValue) is given by the 90th
    # quantile of the absolute value of the gradient.
    print'Computing lambda = q-q-based nonlinear filtering threshold'
    slopeMagnitudeDemArray = slopeMagnitudeDemArray.flatten()
    slopeMagnitudeDemArray = slopeMagnitudeDemArray[~np.isnan(slopeMagnitudeDemArray)]
    
    print 'dem smoothing Quantile', default.edge_thresh
    edgeThresholdValue = np.asscalar(mquantiles(np.absolute(slopeMagnitudeDemArray), default.edge_thresh))
    print 'edgeThresholdValue:', edgeThresholdValue
    kappa = edgeThresholdValue 
    gamma = 0.1
    step = (pixel_res, pixel_res)
    option = 2
    
    img = dem_in.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
    for ii in xrange(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        if option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
        elif option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't ask questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
        # update the image
        mNS = np.isnan(NS)
        mEW = np.isnan(EW)
        NS[mNS] = 0
        EW[mEW] = 0
        NS += EW
        mNS &= mEW
        NS[mNS] = np.nan
        imgout += gamma*NS

    dem_pm_arr = imgout
    end_t = time.time()
    print "Perona-Malik filtering complete, execution time: %.2f" %(end_t - start_t)
    return dem_pm_arr
#def pm_filt_cv(dem_arr, dem_meta, niter = default.pm_n_iter):
#    
#    """ !!! perona malik code is executed using pygeonet code: anisodiff and lambda_nonlinear_filter!!! 
#    source: Sangireddy et al., 2016 http://dx.doi.org/10.1016/j.envsoft.2016.04.026"""
#    
#    #passing masked dem and nan dem yield same results, but masked dem takes 4X as long...extra step of filling clean masked dem with NaNs
#    print "Beginning perona malik filtering with %d iterations..." %(niter)
#    start_t = time.time()
#    pixel_res = float(dem_meta['pix_res'])
#    dem_in = np.ma.filled(dem_arr, np.nan)
#    
#    #perform pygeonet lambda_nonlinear_filter
#    slopeXArray, slopeYArray = np.gradient(dem_in, pixel_res)
#    slopeMagnitudeDemArray = np.sqrt(slopeXArray**2 + slopeYArray**2)
#    print 'DEM slope array shape:', slopeMagnitudeDemArray.shape
#    # plot the slope DEM array
#    # Computation of the threshold lambda used in Perona-Malik nonlinear
#    # filtering. The value of lambda (=edgeThresholdValue) is given by the 90th
#    # quantile of the absolute value of the gradient.
#    print'Computing lambda = q-q-based nonlinear filtering threshold'
#    slopeMagnitudeDemArray = slopeMagnitudeDemArray.flatten()
#    slopeMagnitudeDemArray = slopeMagnitudeDemArray[~np.isnan(slopeMagnitudeDemArray)]
#    
#    ##following is only used to assign curvature as edge stoppping geo characteristic
#    gradXArrayT = slopeXArray
#    gradYArrayT = slopeYArray
#
#    gradGradXArray,tmpy = np.gradient(gradXArrayT,pixel_res)
#    tmpx,gradGradYArray = np.gradient(gradYArrayT,pixel_res)
#    curvatureDemArray = gradGradXArray + gradGradYArray
#    curvatureDemArray[np.isnan(curvatureDemArray)] = 0
#    slopeMagnitudeDemArray = curvatureDemArray
#    del tmpy, tmpx
#    #
#    print 'dem smoothing Quantile', default.edge_thresh
#    edgeThresholdValue = np.asscalar(mquantiles(np.absolute(slopeMagnitudeDemArray), default.edge_thresh))
#    print 'edgeThresholdValue:', edgeThresholdValue
#    kappa = edgeThresholdValue 
#    gamma = 0.1
#    step = (pixel_res, pixel_res)
#    option = 2
#    
#    img = dem_in.astype('float32')
#    imgout = img.copy()
#
#    # initialize some internal variables
#    deltaS = np.zeros_like(imgout)
#    deltaE = deltaS.copy()
#    NS = deltaS.copy()
#    EW = deltaS.copy()
#    gS = np.ones_like(imgout)
#    gE = gS.copy()
#    for ii in xrange(niter):
#        # calculate the diffs
#        deltaS[:-1, :] = np.diff(imgout, axis=0)
#        deltaE[:, :-1] = np.diff(imgout, axis=1)
#        if option == 2:
#            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
#            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
#        elif option == 1:
#            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
#            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
#        # update matrices
#        E = gE*deltaE
#        S = gS*deltaS
#        # subtract a copy that has been shifted 'North/West' by one
#        # pixel. don't ask questions. just do it. trust me.
#        NS[:] = S
#        EW[:] = E
#        NS[1:, :] -= S[:-1, :]
#        EW[:, 1:] -= E[:, :-1]
#        # update the image
#        mNS = np.isnan(NS)
#        mEW = np.isnan(EW)
#        NS[mNS] = 0
#        EW[mEW] = 0
#        NS += EW
#        mNS &= mEW
#        NS[mNS] = np.nan
#        imgout += gamma*NS
#
#    dem_pm_arr = imgout
#    
#    end_t = time.time()
#    print "Perona-Malik filtering complete, execution time: %.2f" %(end_t - start_t)
#    return dem_pm_arr

def main(dem_arr, dem_meta, smoothing_width = default.smoothing_width):
    
    #perform filtering - should pass the masked, clean dem to filter modules*
    filt_window = calc_filt_params(dem_meta, smoothing_width)
    dem_mean = mean_filt(dem_arr, filt_window)
    dem_med = med_filt(dem_arr, filt_window)
    dem_gaus = gaus_filt(dem_arr)         
    dem_pm_slp = pm_filt_slp(dem_arr, dem_meta)
#    dem_pm_cv = pm_filt_cv(dem_arr, dem_meta)

    dem_mean_tif = array_to_geotif(dem_mean, dem_meta, default.roi_dems, default.dem_mean)
    dem_med_tif = array_to_geotif(dem_med, dem_meta, default.roi_dems, default.dem_med)
    dem_gaus_tif = array_to_geotif(dem_gaus, dem_meta, default.roi_dems, default.dem_gaus)
    dem_pmslp_tif = array_to_geotif(dem_pm_slp, dem_meta, default.roi_dems, default.dem_pm_slp)  
#    dem_pmcv_tif = array_to_geotif(dem_pm_cv, dem_meta, default.roi_dems, default.dem_pm_cv)   
    
    return dem_mean_tif, dem_med_tif, dem_gaus_tif, dem_pmslp_tif


if __name__== '__main__':
    main()
    sys.exit(0)

    