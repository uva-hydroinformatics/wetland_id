"""
Image directory home >> .\tensorflow\models\research\deeplab\datasets\wetlands
    +dataset
        +ImageSets --> contains train.txt, val.txt, trainval.txt
        +JPEGImages --> input color images (data), *.jpg
        +SegmentationClass --> ground truth annotations (wetland/BG binary) corresponding to each JPEGImage
    +tfrecord

Author: Gina O'Neil
Initial Version: Feb. 13, 2019
"""

from osgeo import gdal, ogr
import os
import sys
import subprocess
import glob
import shutil
import numpy as np
import raster_array_funcspy35 as ra
import shlex
import math
from random import shuffle
import sklearn.preprocessing as skp

import matplotlib.pyplot as plt

def scale(band_orig, band_min, band_max, no_data):
    #scale 0 to 1
    band_masked = np.ma.masked_values(band_orig, no_data)
    band_masked[(~band_masked.mask) & (band_masked < band_min)] = band_min
    band_masked[(~band_masked.mask) & (band_masked > band_max)] = band_max
    
    band_nan = np.ma.filled(band_masked, np.nan)  
    scaled_band = (band_nan - band_min) / (band_max - band_min)    
    band_masked = np.ma.masked_invalid(scaled_band)    

    return band_masked

def build_imgs(tif_in, wetlands_shp, bounds_shp, img_dir, tilesize, no_data_val, train_percent, site_name):
    """
    :param tif_in: geotiff with n bands, each band representing input data...variable reset to the scaled tif (0-1)
    :param wetlands_shp: verification wetlands shapefile
    :param bounds_shp: limits of verification wetlands
    :param img_dir: ./datasets/wetlands/dataset
    :param tilesize: dimensions of images (pixels)
    :param no_data_val: no data value to assign to output images, will be same for input data images and annotations
    :return: text file containing list of eligible images to use in train/test split
    """
    #based on distribution
    band1_min = 0.00001
    band1_max = 30. 
    band2_min = -3.
    band2_max = 3. 
    band3_min = 0.00001
    band3_max = 35.
    band4_min = -1.
    band4_max = 1. 

    nonw_val = 1
    w_val = 2

    #define file path variables
    ImageSets = os.path.join(img_dir, "ImageSets")
    IMG_Tiles = os.path.join(img_dir, "Imgs")
    GT_Tiles = os.path.join(img_dir, "Verif")
    eligImg = os.path.join(ImageSets, 'trainval.txt')
    trainImg = os.path.join(ImageSets, 'train.txt')
    valImg = os.path.join(ImageSets, 'val.txt')
    extraImg = os.path.join(ImageSets, 'extra.txt')
    
    #set up directories
    dirs = [ImageSets, IMG_Tiles, GT_Tiles]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            
    if os.path.exists(eligImg):
        os.remove(eligImg)

    #open orig geotiff and get info
    dset, meta = ra.gtiff_to_arr(tif_in, 'float')
    
    #create empty array with shape of composite tif
    scaled_arr = np.ndarray(shape=np.shape(dset))

    band1 = dset[:, :, 0] #TWI
    band2 = dset[:, :, 1] #curv
    band3 = dset[:, :, 2] #dtw
    band4 = dset[:, :, 3] #NVDI
    band4[np.isnan(band4)] = -9999.

    scaled_band1 = scale(band1, band_min=band1_min, band_max=band1_max, no_data=-9999)
    scaled_band2 = scale(band2, band_min=band2_min, band_max=band2_max, no_data=-9999)
    scaled_band3 = scale(band3, band_min=band3_min, band_max=band3_max, no_data=-9999)
    scaled_band4 = scale(band4, band_min=band4_min, band_max=band4_max, no_data=-9999)
   
    scaled_arr[:, :, 0] = np.ma.filled(scaled_band1, no_data_val)
    scaled_arr[:, :, 1] = np.ma.filled(scaled_band2, no_data_val)
    scaled_arr[:, :, 2] = np.ma.filled(scaled_band3, no_data_val)
    scaled_arr[:, :, 3] = np.ma.filled(scaled_band4, no_data_val)
        
    tif_in = ra.arr_to_gtiff(scaled_arr, meta, img_dir, "scaled_data.tif", nodata=no_data_val)

    #rasterize wetland and nonwetland area    
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(bounds_shp, 0)
    inLayer = inDataSource.GetLayer()
    bounds_ext = inLayer.GetExtent()
    
    gt_tif_temp = os.path.join(img_dir, "annotationTEMP.tif")

    list_options = [
        '-init {}'.format(nonw_val),
        '-burn {}'.format(w_val),
        #'-a_nodata -1',#{}'.format(no_data_val),
        '-ot Int16',
        '-co COMPRESS=LZW',
        '-te {} {} {} {}'.format(bounds_ext[0], bounds_ext[2], bounds_ext[1], bounds_ext[3]),
        '-tr {} {}'.format(meta['pix_res'], meta['pix_res'])
    ]
    
    options_string = " ".join(list_options)
    gdal.Rasterize(gt_tif_temp, wetlands_shp, options=options_string)
    
    #get correct no data vals
    gt_tif = ra.clip_geotif(gt_tif_temp, img_dir, bounds_shp, no_data=no_data_val)
    os.remove(gt_tif_temp)
    width = meta['ncol']
    height = meta['nrow']

    #first image #
    img_n = 0

    #cut composite.tif into tilesize x tilesize tiles
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j

            #clip data geotiff to extents and save as "[IMG#].tif" in imgdir
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + tif_in + " " + IMG_Tiles + r"\IMG{}.tif".format(img_n)

            subprocess.call(cmd, shell=True)

            #clip annotation geotiff to extents and save as "[IMG#].tif" in imgdir            
            cmd = "gdal_translate -of GTIFF -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + gt_tif + " " + GT_Tiles + r"\IMG{}.tif".format(img_n)

            subprocess.call(cmd, shell=True)
            
            #increase img #
            img_n +=1
            
    extra = []
    eligible_imgs = []
    img_count = 0
    
    for f_i in os.listdir(IMG_Tiles):

        img_in = os.path.join(IMG_Tiles, f_i)
        img_name = f_i[:-4]

        dset_tile, meta = ra.gtiff_to_arr(img_in, 'float')
        
        gt_in = os.path.join(GT_Tiles, f_i)
        gt_tile, _ = ra.gtiff_to_arr(gt_in, 'byte')

        #check for dimensions
        if (meta['ncol'] == tilesize) & (meta['nrow'] == tilesize):
  
            #check for tiles with at least 20% real data
            if (gt_tile == 1).sum() + (gt_tile == 2).sum() > (0.2 * gt_tile.size):    
                # write filename to trainval.txt
                img_count = img_count + 1
                eligible_imgs.append(img_name)
    
            else:
                print("More than 20% of the area was no data, {} was rejected for training and testing.".format(f_i))
                extra.append("{}_{}".format(site_name, img_name))      
        else:
            print("{} was rejected for training and testing due to dimensions of {}.".format(f_i, np.shape(gt_tile)))
            extra.append("{}_{}".format(site_name, img_name))

    #perform train/test split and write to files

    val_percent = 1 - (train_percent)

    with open(eligImg, "w") as text:
        text.writelines("\n".join(e for e in eligible_imgs)) 
    text.close()

    shuffle(eligible_imgs)
    
    trainImgThresh = math.ceil(len(eligible_imgs) * train_percent) #sample training only from images with sufficient data
    
    #random selection of training images, use rest for testing
    trainList = eligible_imgs[: trainImgThresh]
    valImgThresh = int(len(eligible_imgs) * val_percent)
    valList = eligible_imgs[ -valImgThresh : ]
        
    #add in site name
    trainList_info = ["{}_{}".format(site_name, t) for t in trainList]
    
    #write "site name, img name" to file
    with open(trainImg, "w") as text:
        text.writelines("\n".join(t for t in trainList_info))
    text.close()

    valList_info = ["{}_{}".format(site_name, v) for v in valList]
    
    with open(valImg, "w") as text:
        text.writelines("\n".join(v for v in valList_info))
    text.close()
    
    #append the trejected training tiles for testing later
    with open(extraImg, "w") as text: 
        text.writelines("\n".join(ex for ex in extra))
    text.close()
    
    print("\nStudy site split into {} images\n".format(len(os.listdir(IMG_Tiles))))
    print("{} eligible images with > 20% real data\n".format(img_count))
    print("{} training images and {} testing images\n".format(len(trainList), len(valList)))

    return eligImg, trainList, valList
