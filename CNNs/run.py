import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler
import torch.nn.init
import os
import pandas as pd
import deepnets_EO as deep
import raster_array_funcspy35 as ra
import create_img_dir2 as cid
import params
import sys
from osgeo import gdal
import numpy as np

"""
CHECK L2 IN DEEPNETS_EO BEFORE RUNNING + CHANNEL NUMBER
"""

#Input Files
tif_in = os.path.join(params.img_dir, "composite_4.tif")
wetlands_shp = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\wetlands.shp".format(params.site_name)
bounds_shp = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\lims_prj.shp".format(params.site_name)

#input()
#outdir= r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\Site4"
#dem = r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\Site4\NVDI.tif"
#arr, meta = ra.gtiff_to_arr(dem, "float")
#print(np.min(arr))
#arr[arr==-9999.] = np.nan
#print(np.min(arr))
#
#arr_ma = np.ma.masked_invalid(arr)
#print(np.min(arr_ma))
#
#arr_out = np.ma.filled(arr_ma, -9999.)
#print(np.min(arr_out), np.max(arr_out))
#arr_tif = ra.arr_to_gtiff(arr, meta, outdir, "pm50DEM_fix.tif")
#sys.exit(0)

#filepaths
img_dir = params.img_dir
ImageSets = params.ImageSets 
IMG_Tiles = params.IMG_Tiles
GT_Tiles = params.GT_Tiles 

outdir = os.path.join(img_dir, params.run_name)


if not os.path.exists(outdir):
    os.mkdir(outdir)
    

#******************************Create Image dataset***************************#

if not os.path.exists(ImageSets):
    eligble_images, trainList, valList = \
    cid.build_imgs(tif_in, wetlands_shp, bounds_shp, img_dir, \
                   tilesize=params.img_tile_size, no_data_val=params.nd_val, \
                   train_percent=params.train_percent, site_name=params.site_name) 

#**************************Begin DeepNets Execution***************************#

train_sets = params.train_sets
train_ids_full = []
all_imgs = []

for ts in train_sets:
    current = [line.rstrip('\n') for line in open(ts, "r")]
    train_ids_full.extend(current)   
    all_imgs.extend(current)
    
if params.train_thresh is not None:
    print("Only using first {} training images".format(params.train_thresh))
    train_ids = train_ids_full[:params.train_thresh]
    add_on = train_ids_full[params.train_thresh:]
else:
    print("Using full training sets......................")
    train_ids = train_ids_full
    add_on = []

test_sets = params.test_sets
test_ids_full = []
add_on = train_ids_full #uncomment if using BOTH train and test images to evaluate a model
for ts in test_sets:    
    current = [line.rstrip('\n') for line in open(ts, "r")]
    test_ids_full.extend(current)
    all_imgs.extend(current)

#check for errors in image dataset creations
for t in train_ids_full:
    if t in test_ids_full:
        sys.exit("Stop, check image directories, {} was found in training and testing sets".format(t))

test_ids =  test_ids_full
for i in add_on:
    test_ids.append(i)

##uncomment for models trained with LIMITED number of images from multiple sites
#add_on = []
#for ts in train_sets:
#    current = [line.rstrip('\n') for line in open(ts, "r")]
#    train_ids_full.extend(current[:params.train_thresh])      
#    all_imgs.extend(current)
#    add_on.extend(current[params.train_thresh:]) 
#print(train_ids_full)
#print(add_on)
#train_ids = train_ids_full
#
#test_sets = params.test_sets
#test_ids_full = []
#
#for ts in test_sets:    
#    current = [line.rstrip('\n') for line in open(ts, "r")]
#    test_ids_full.extend(current)
#    all_imgs.extend(current)
#
##check for errors in image dataset creations
#for t in train_ids_full:
#    if t in test_ids_full:
#        sys.exit("Stop, check image directories, {} was found in training and testing sets".format(t))
#
#test_ids =  test_ids_full
#for i in add_on:
#    test_ids.append(i)


#start deep nets
net, sched, optimizer = deep.init_model()

#load data
train_loader = deep.load_data(train_ids)

#train the model    
#final_model = deep.train(net, optimizer, params.epochs, train_loader, test_ids, \
#                         savedir = outdir, run = params.run_name, scheduler = sched)
#final_model = os.path.join(outdir, "segnet_final")
final_model = r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\Site3\run_G\run_G_epoch100_p0.199_r0.864"
print(final_model)


#load the model
net.load_state_dict(torch.load(final_model))


#test the model
_, all_preds, all_gts, cm_pd, class_report_df, acc_scores_all, iou_pd = \
                        deep.test(net, test_ids, all=True)

#save predictions to geotiffs
#for p, id_ in zip(all_preds, test_ids):
#    #get the georeferenced data from the gt image
#    site_n, img_id = id_.split("_")
#    gt_src = os.path.join(r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\{}\Verif".format(site_n), "{}.tif".format(img_id))
#    src_arr, src_meta = ra.gtiff_to_arr(gt_src, dtype='int')
#    geotiff = ra.arr_to_gtiff(p, src_meta, outdir, '100TEST_inference_tile_{}.tif'.format(id_), dtype='int', nodata=0)
#    plt.imshow(p)
    
#save accuracy
acc_report = "{}_TEST100_accuracy.xlsx".format(params.run_name)
writer=pd.ExcelWriter(os.path.join(outdir, acc_report))
cm_pd.to_excel(writer, sheet_name = 'Output_accuracy_metrics', startrow=0)
class_report_df.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=True, startrow=6)
acc_scores_all.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=False, startrow=12)   
iou_pd.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=False, startrow=14)  
writer.save()


#run model for training image set, save training results to geotiffs
_, all_preds, all_gts, cm_pd, class_report_df, acc_scores_all, iou_pd = \
                    deep.test(net, train_ids, all=True)

#save predictions to geotiffs
#for p, id_ in zip(all_preds, train_ids):
#    #get the georeferenced data from the gt image
#    site_n, img_id = id_.split("_")
#    gt_src = os.path.join(r"D:\3rdStudy_ONeil\wetland_id\CNNs\pytorch\dataset\{}\Verif".format(site_n), "{}.tif".format(img_id))
#    src_arr, src_meta = ra.gtiff_to_arr(gt_src, dtype='int')
#    geotiff = ra.arr_to_gtiff(p, src_meta, outdir, '100TRAIN_inference_tile_{}.tif'.format(id_), dtype='int', nodata=0)
#    plt.imshow(p)

##save accuracy
#acc_report = "{}_TRAIN100_accuracy.xlsx".format(params.run_name)
#writer=pd.ExcelWriter(os.path.join(outdir, acc_report))
#cm_pd.to_excel(writer, sheet_name = 'Output_accuracy_metrics', startrow=0)
#class_report_df.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=True, startrow=6)
#acc_scores_all.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=False, startrow=12)   
#iou_pd.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=False, startrow=14)  
#writer.save()
