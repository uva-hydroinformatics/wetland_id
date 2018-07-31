from osgeo import gdal, gdal_array, ogr
import numpy as np
import pandas as pd
import subprocess
import sys
import wetland_id_defaults as default
import scipy
from scipy import stats
import os
import time

def geotif_to_array(path_to_file):
    if path_to_file[-3:] != "tif":
        print "Wetland ID tool is only configured to process geotiffs. \n"
        print  "exiting program... \n"
        sys.exit(0)
    else:
        print "Reading in %s as an array..." %(path_to_file) + '\n'
		#TODO: look into benefits of switch to xarray dataframe stucture 

		#get geotiff metadata    
        tif_ds = gdal.Open(os.path.join(path_to_file), gdal.GA_ReadOnly)
		
        driver = tif_ds.GetDriver()
		
        prj = tif_ds.GetProjection()
		
        ncol = tif_ds.RasterXSize
		
        nrow = tif_ds.RasterYSize
		
        ext = tif_ds.GetGeoTransform()
		
        n_bands = tif_ds.RasterCount
		
        pixel_res = ext[1]

		#NOTE: all tiffs must be read in as float arrays in order to set missing values to np.nan \
		#this could be changed if there is a method to create geotiffs such that masked elements are NaN
		
		#prepare empty array with target size tif_ds.GetRasterBand(1).DataType
		
        tif_as_array = np.zeros((tif_ds.RasterYSize, tif_ds.RasterXSize, tif_ds.RasterCount), \
					   gdal_array.GDALTypeCodeToNumericTypeCode(gdal.GDT_Float32))
					   
		
        print 'Array created from %s has shape:' %(path_to_file) 
		
        print tif_as_array.shape, '\n'
		
		#populate the empty array

        if n_bands > 1:		
            for b in range(tif_as_array.shape[2]):
                tif_as_array[:, :, b] = tif_ds.GetRasterBand(b + 1).ReadAsArray()    
        else:
            tif_as_array[:,:,0] = tif_ds.GetRasterBand(1).ReadAsArray()
            tif_as_array = tif_as_array[:,:,0]

            #save tiff meta data
        
        tif_meta = { 'driver' : driver, 'prj' : prj, 'ncol' : ncol, 'nrow' : nrow, 'ext' : ext, 'nbands' : n_bands, 'pix_res' : pixel_res }

    return tif_as_array, tif_meta

def array_to_geotif(data, data_meta, fpath, filename, dtype='float', nodata=-9999):
    """
    data_meta = dict-like georeferencing information needed to write new geotiff
    """
    #NOTE: pass data_meta that matches the desired shape (2d or 3d)
    print "Writing array to geotiff: %s..."%(filename) + '\n'
    if dtype == 'float':
        gdal_type = gdal.GDT_Float32
    elif dtype == 'int':
        gdal_type = gdal.GDT_Int32
    else:
        sys.exit("Datatype not recognized, system exiting.....")

    saveas = os.path.join(fpath, filename)
    driver = data_meta['driver']
    ncol, nrow = data_meta['ncol'], data_meta['nrow']
    prj = data_meta['prj']
    ext = data_meta['ext']
    n_bands = data_meta['nbands']
    out_raster_ds = driver.Create(saveas, ncol, nrow, n_bands, gdal_type, ['COMPRESS=LZW'])
    out_raster_ds.SetProjection(prj)
    out_raster_ds.SetGeoTransform(ext)
        
    if n_bands > 1:
        for b in range(n_bands):
            out_raster_ds.GetRasterBand(b + 1).WriteArray(data[:, :, b])
            band = out_raster_ds.GetRasterBand(b + 1)
            band.SetNoDataValue(nodata)
    else:
        out_raster_ds.GetRasterBand(1).WriteArray(data)
        band = out_raster_ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
	# Close dataset
    out_raster_ds = None
    
    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(saveas) 
    subprocess.call(cmd_info, shell = True)
    return saveas

def clean_array(arr):
    #HACK: dynamically find nan value by taking mode of corner values
    if np.ndim(arr) > 2:
        nan_val_list = [arr[0,0,0], arr[-1,-1,0], arr[0,-1,0], arr[-1,0,0] ]
        nan_val_mode = stats.mode(nan_val_list, axis=0)
        nan_val = nan_val_mode[0].item()
        print "Detected %f to be a NaN Value." %(nan_val)
        tif_arr_mask = np.ma.masked_values(arr, nan_val)
#        if np.isnan(np.min(arr)):
#            print "yes"
#            tif_arr_mask1 = np.ma.masked_where(np.isnan(arr), arr)
#            tif_arr_mask = np.array(tif_arr_mask1)
#            tif_arr_mask[np.isnan(tif_arr_mask)]= -9999
##            clean_array(tif_arr_mask)
#        else:
#            tif_arr_mask = np.ma.masked_values(arr, nan_val)
    else:
        nan_val_list = [arr[0,0], arr[-1,-1], arr[0,-1], arr[-1,0] ]
        nan_val_mode = stats.mode(nan_val_list, axis=0)
        nan_val = nan_val_mode[0].item()
        print "Detected %f to be a NaN Value." %(nan_val)
        tif_arr_mask = np.ma.masked_values(arr, nan_val)
#        if np.isnan(np.min(arr)):
#            print "yes"
#            tif_arr_mask1 = np.ma.masked_where(np.isnan(arr), arr)
#            tif_arr_mask = np.array(tif_arr_mask1)
#            tif_arr_mask[np.isnan(tif_arr_mask)]= -9999
##            clean_array(tif_arr_mask)
#        else:
#            tif_arr_mask = np.ma.masked_values(arr, nan_val)
#    print "Detected %f to be a NaN Value, Continue? " %(nan_val)
#    print "Enter 0 to continue, 1 to exit program ..."
#    t_end = time.time() + 60
#    while time.time() < t_end:
#        test = input()
#        if test == 1:
#            sys.exit(0)
#            break
#        else:
#            break
    
    ##mask nan values
#    tif_arr_mask = np.ma.masked_equal(arr, nan_val)
#    tif_arr_mask = np.ma.masked_values(arr, nan_val)
    tif_arr_clean = tif_arr_mask.reshape(np.shape(arr))    
    
    return tif_arr_clean

def clip_geotif(tif_in, out_tif_path, clip_bounds):
    tif_ds = gdal.Open(tif_in, gdal.GA_ReadOnly)
    tif_name = tif_in.split('\\')[-1]
    ext = tif_ds.GetGeoTransform()
    pix_res = float(ext[1])
    tif_out = os.path.join(out_tif_path, tif_name[:-4] + default.clip_suf)
    cmd = "gdalwarp.exe -cutline \"%s\" -dstnodata -9999. -tr %f %f -overwrite -r bilinear \
        -crop_to_cutline \"%s\" \"%s\" -co COMPRESS=LZW" %(clip_bounds, pix_res, pix_res, tif_in, tif_out) 
    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(tif_out)
    subprocess.call(cmd)
    subprocess.call(cmd_info)
    print "%s has been clipped!" %(tif_in)
    return tif_out

def rasterize_simple(shp_in, out_tif_path, out_tif_name, out_tif_val, pix_res):  
    tif_out = os.path.join(out_tif_path, out_tif_name)
    
    cmd = "gdal_rasterize -burn %f -a_nodata -9999. -ot Float32 -tr %f %f %s %s" \
    %(out_tif_val, pix_res, pix_res, shp_in, tif_out)
    
    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(tif_out) 
    
    subprocess.call(cmd, shell = True)
    subprocess.call(cmd_info, shell = True)
    
    print "%s has been created! \n" %(tif_out)
    
    return tif_out

def rasterize_opts(shp_in, out_tif_path, out_tif_name, out_tif_val, pix_res, ext, outter_vals):
    #ext should be a list of xmin, ymin, xmax, ymax or None to assign minmum extent based on shape
    
    tif_out = os.path.join(out_tif_path, out_tif_name)
    
    cmd = "gdal_rasterize -init %f -burn %f -a_nodata -9999. -ot Float32 -co COMPRESS=LZW \
    -te %f %f %f %f -tr %f %f %s %s" \
    %(outter_vals, out_tif_val, ext[0], ext[2], ext[1], ext[3], pix_res, pix_res, shp_in, tif_out)

    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(tif_out) 
    
    subprocess.call(cmd, shell = True)
    subprocess.call(cmd_info, shell = True)
    
    print "%s has been created! \n" %(tif_out)
    
    return tif_out
    
def create_verif(wetlands_shp, bounds_shp, out_tif_path, pix_res):
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(bounds_shp, 0)
    inLayer = inDataSource.GetLayer()
    bounds_ext = inLayer.GetExtent()
    verif_tif = rasterize_opts(wetlands_shp, out_tif_path, "verif.tif", 0., pix_res, bounds_ext, 1.)
    return verif_tif
