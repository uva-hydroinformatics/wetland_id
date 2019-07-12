"""
Functions for georeferenced gridded data I/O. Configured to read geotiffs in as ndarrays
and print ndarrays to geotiffs. Geotiffs can be converted to PNG and JPEG images.

Author: Gina O'Neil
"""

from osgeo import gdal, gdal_array, ogr
import numpy as np
import subprocess
import sys
import scipy
from scipy import stats
import os
import time

def gtiff_to_arr(path_to_file, dtype):
    """
    :param path_to_file: filepath to input geotiff
    :param dtype: datatype of pixels
    :return: ndarray
    """

    if path_to_file[-3:] != "tif":
        print ("Wetland ID tool is only configured to process geotiffs. \n")
        return None

    else:
        print ("Reading in %s as an array..." %(os.path.basename(path_to_file) + '\n'))
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

        if dtype == 'float':
            gdal_type = gdal.GDT_Float32
        elif dtype == 'int':
            gdal_type = gdal.GDT_Int32
        elif dtype == 'byte':
            gdal_type = gdal.GDT_Byte

        tif_as_array = np.zeros((tif_ds.RasterYSize, tif_ds.RasterXSize, tif_ds.RasterCount), \
                       gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


        print ('Array created from %s has shape:' %(os.path.basename(path_to_file)))
        print (tif_as_array.shape, '\n')

        #populate the empty array
        if n_bands > 1:
            for b in range(tif_as_array.shape[2]):
                tif_as_array[:, :, b] = tif_ds.GetRasterBand(b + 1).ReadAsArray()
        else:
            tif_as_array[:,:,0] = tif_ds.GetRasterBand(1).ReadAsArray()
            tif_as_array = tif_as_array[:,:,0]

            #save tiff meta data
        tif_meta = { 'driver' : driver, 'prj' : prj, 'ncol' : ncol, 'nrow' : nrow, 'ext' : ext, 'nbands' : n_bands, 'pix_res' : pixel_res }
    tif_ds = None

    return tif_as_array, tif_meta

def arr_to_gtiff(data, data_meta, fpath, fname, dtype='float', nodata=-9999):
    """
    :param data: ndarray
    :param data_meta: (dict) georeferenced meta data for ndarray
    :param fpath: output path
    :param fname: output filename
    :param dtype: target gdal data type
    :param nodata: gdal no data value
    :return: file path to output geotiff
   """

    print ("Writing array to geotiff: %s..."%(fname) + '\n')
    if dtype == 'float':
        gdal_type = gdal.GDT_Float32
    elif dtype == 'int':
        gdal_type = gdal.GDT_Int32
    elif dtype == 'byte':
        gdal_type = gdal.GDT_Byte
    else:
        sys.exit("Datatype not recognized, system exiting.....")

    saveas = os.path.join(fpath, fname)
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
    subprocess.call(cmd_info, shell = False)
    return saveas

def gtiff_to_img(tif_in, fpath, fname, img_type, no_data_val):
    """
    :param tif_in: filepath to geotiff to be converted
    :param fpath: (str) img out file path
    :param fname: (str) img out filename
    :param img_type: (str) "JPG" or "PNG", n_bands > 1 should use JPG
    :return: filepath to new img
    """
    imgout = os.path.join(fpath, fname)

    if img_type == "JPG":
        list_options = [
            '-ot Byte',
            '-of JPEG',
            '-scale',  # inputs are scaled to 0-255
            '-co QUALITY=100 TILED=YES'#,
            #'-a_nodata {}'.format(no_data_val)
        ]
        options_string = " ".join(list_options)

    elif img_type == "PNG":
        list_options = [
            '-ot Byte',
            '-of PNG'#,
            # '-scale 0 1 1 2',  # change here to assign different values to gt classes
            #'-a_nodata {}'.format(no_data_val)
        ]
        options_string = " ".join(list_options)

    else:
        print ("Only JPG or PNG images can be created.")
        return ""

    gdal.Translate(imgout, tif_in, options=options_string)

    print ("Converted {} to {} image!".format(os.path.basename(tif_in), img_type))

    return imgout

# TODO: def array_to_img(data, datameta, fpath, fname, dtype='float', nodata=-9999, ext):

def clean_array(arr, no_data=None):
    """
    :param arr: Ndarray
    :param no_data: no data value if known, otherwise will be guessed by taking mode of corner values
    :return: clean array where no data values are masked
    """
    if no_data==None:
        if np.ndim(arr) > 2:
            nan_val_list = [arr[0,0,0], arr[-1,-1,0], arr[0,-1,0], arr[-1,0,0] ]
            nan_val_mode = stats.mode(nan_val_list, axis=0)
            nan_val = nan_val_mode[0].item()
            print ("Detected %f to be a NaN Value." %(nan_val))
            tif_arr_mask = np.ma.masked_values(arr, nan_val)

        else:
            nan_val_list = [arr[0,0], arr[-1,-1], arr[0,-1], arr[-1,0] ]
            nan_val_mode = stats.mode(nan_val_list, axis=0)
            nan_val = nan_val_mode[0].item()
            print ("Detected %f to be a NaN Value." %(nan_val))
            tif_arr_mask = np.ma.masked_values(arr, nan_val)
    else:
        nan_val = no_data
        tif_arr_mask = np.ma.masked_values(arr, nan_val)

    tif_arr_clean = tif_arr_mask.reshape(np.shape(arr))

    return tif_arr_clean

def clip_geotif(tif_in, fpath, clip_bounds, suf = "_c.tif", no_data=-9999.):
    """
    :param tif_in: input geotif
    :param fpath: output filepath
    :param clip_bounds: shapefile to use as clipping bounds
    :param suf: suffix to add to tif_in base name
    :param no_data: optional no data value
    :return: filepath to clipped geotif
    """
    tif_ds = gdal.Open(tif_in, gdal.GA_ReadOnly)
    tif_name = os.path.basename(tif_in)
    ext = tif_ds.GetGeoTransform()
    pix_res = float(ext[1])
    tif_out = os.path.join(fpath, tif_name[:-4] + suf)

    cmd = "gdalwarp.exe -cutline \"%s\" -dstnodata %d -tr %f %f -overwrite -r bilinear \
        -crop_to_cutline \"%s\" \"%s\"" %(clip_bounds, no_data, pix_res, pix_res, tif_in, tif_out)

    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(tif_out)

    subprocess.call(cmd)
    subprocess.call(cmd_info)

    print ("%s has been clipped!" %(tif_name))
    return tif_out

def rasterize_simple(shp_in, fpath, fname, out_tif_val, pix_res):
    """
    :param shp_in: shapefile to be rasterised
    :param fpath: output filepath
    :param fname: output fileanme
    :param out_tif_val: value to burn into new raster
    :param pix_res: output pixel resolution
    :return: filepath to new geotif raster
    """

    tif_out = os.path.join(fpath, fname)

    cmd = "gdal_rasterize -burn %f -a_nodata -9999. -ot Float32 -tr %f %f %s %s" \
    %(out_tif_val, pix_res, pix_res, shp_in, tif_out)

    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(tif_out)

    subprocess.call(cmd, shell = True)
    subprocess.call(cmd_info, shell = True)

    print ("%s has been created! \n" %(tif_out))

    return tif_out

def rasterize_opts(shp_in, fpath, fname, out_tif_val, pix_res, ext, outter_vals):
    """
    :param shp_in: shapefile to be rasterised
    :param fpath: output filepath
    :param fname: output fileanme
    :param out_tif_val: value to burn into new raster
    :param pix_res: output pixel resolution
    :param ext: ext of output raster, list of xmin, ymin, xmax, ymax
    :param outter_vals: pixel values for pixels outside of shapefile but within extents
    :return: filepath to new geotif raster
    """

    tif_out = os.path.join(fpath, fname)

    cmd = "gdal_rasterize -init %f -burn %f -a_nodata -9999. -ot Float32 -co COMPRESS=LZW -te %f %f %f %f -tr %f %f %s %s" \
    %(outter_vals, out_tif_val, ext[0], ext[2], ext[1], ext[3], pix_res, pix_res, shp_in, tif_out)
    print(cmd)

    cmd_info = 'gdalinfo.exe -stats \"%s\"'%(tif_out)

    subprocess.call(cmd, shell = True)
    subprocess.call(cmd_info, shell = True)

    print ("%s has been created! \n" %(tif_out))

    return tif_out

def create_verif(wetlands_shp, bounds_shp, fpath, pix_res):
    """
    :param wetlands_shp: wetlands shapefile
    :param bounds_shp: limits shapefile
    :param fpath: output filepath
    :param pix_res: output pixel resolution
    :return: filepath to new verification raster
    """

    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(bounds_shp, 0)
    inLayer = inDataSource.GetLayer()
    bounds_ext = inLayer.GetExtent()
    verif_tif = rasterize_opts(wetlands_shp, fpath, "verif.tif", 0., pix_res, bounds_ext, 1.)

    return verif_tif
