import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler
import torch.nn.init
import os
import pandas as pd
import deepnets_EO as deep
import raster_array_funcspy35 as ra
import create_img_dir as cid
#import params
import sys
import numpy as np

run_name = "run_G"
site_name = "Site4"
img_dir = os.path.join(r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset", site_name) 
#Input Files
#tif_in = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\composites\comp_o_c.tif".format(params.site_name)
#tif_in = os.path.join(img_dir, "composite_4.tif")
tif_in = r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\Site4\NAIP_c.tif"
wetlands_shp = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\wetlands.shp".format(site_name)
bounds_shp = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\lims_prj.shp".format(site_name)
#NAIP_in = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\NAIP.tif".format(site_name)
NAIP_in = tif_in

#outdir = os.path.join(img_dir, run_name)
outdir = r"D:\3rdStudy_ONeil\Results\RF\Site4"
if not os.path.exists(outdir):
    os.mkdir(outdir)

#
#NAIP = ra.clip_geotif(NAIP_in, img_dir, bounds_shp)
#NAIP = os.path.join(img_dir, "NAIP_c.tif")

NAIP_arr, NAIP_meta = ra.gtiff_to_arr(NAIP_in, dtype="float")

def calc_NDVI(NAIP_arr):
    
    R = NAIP_arr[:, :, 0]
    NIR = NAIP_arr[:, :, 3]
    NDVI = np.divide((NIR-R), (NIR+R))
    
    return NDVI

NDVI = calc_NDVI(NAIP_arr)


NAIP_meta['nbands'] = 1

NDVI_tiff = ra.arr_to_gtiff(NDVI, NAIP_meta, outdir, "NVDI.tif")

