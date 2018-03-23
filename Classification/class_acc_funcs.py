# -*- coding: utf-8 -*-
"""
This program executes a train/test split, random forest classification, and accuracy assessment.
   Authors:    G. O'Neil and L. Saby
   Changelog: 20180323: Initial version
"""
from sklearn.ensemble import RandomForestClassifier
from osgeo import gdal, gdal_array
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import re
from io import StringIO
import ntpath
import time
import scipy
from scipy import stats
import subprocess

clip_bounds = r"H:\P06_RT29\Data\VDOT_Verification\P06_LimitsP.shp"
data_dir = r"D:\2ndStudy_ONeil\Terrain_Processing\sklearn_files" 
results_dir = r"D:\2ndStudy_ONeil\Scratch_Output\results" 

"""
Global classification parameters:
    
    w_train_prop, nw_train_prop = proportion of w,nw verified data used to train model, complement /
                of this is automatically reserved to test model
    
    n_trees = number of estimators in RF model
    
    tree_depth = max. depth of each tree
    
    w_cw = class weight assigned to the positive (wetland) class
            positive pixel value = 0
    
    nw_cw = class weight of the negative (nonwetland) class
            negative pixel value = 1
    
"""


n_trees = 300
tree_depth = 50
wcw = 1
nwcw = 1
w_train_prop = 0.1
nw_train_prop = 0.002

def geotiff_to_array(fpath, filename):
    
    print "Reading in %s and converting to an array..." %(filename) + '\n'
    #TODO: look into benefits of switch to xarray dataframe stucture 
    tif_ds = gdal.Open(os.path.join(fpath, filename), gdal.GA_ReadOnly)
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
    tif_arr = np.zeros((tif_ds.RasterYSize, tif_ds.RasterXSize, tif_ds.RasterCount), \
                   gdal_array.GDALTypeCodeToNumericTypeCode(gdal.GDT_Float64))
    print 'Array created from %s has shape:' %(filename) 
    print tif_arr.shape, '\n'
    
    #populate the empty array
    for b in range(tif_arr.shape[2]):
        tif_arr[:, :, b] = tif_ds.GetRasterBand(b + 1).ReadAsArray()    
    
    #save tiff meta data
    tif_meta = { 'driver' : driver, 'prj' : prj, 'ncol' : ncol, 'nrow' : nrow, 'ext' : ext, 'nbands' : n_bands, 'pix_res' : pixel_res }

    return tif_arr, tif_meta
    
    
def clean_data(arr):
    #HACK: dynamically find nan value by taking mode of corner values
    nan_val_list = [arr[0,0,0], arr[-1,-1,0], arr[0,-1,0], arr[-1,0,0] ]
    nan_val_mode = stats.mode(nan_val_list, axis=0)
    nan_val = nan_val_mode[0].item()
    
    #mask nan values
    tif_arr_clean = np.ma.masked_equal(arr, nan_val)
        
    return tif_arr_clean


def array_to_geotiff(fpath, filename, data, data_meta):
    """
    data_meta = dict-like georeferencing information needed to write new geotiff
    """
    
    #NOTE: pass data_meta that matches the desired shape (2d or 3d)
    print "Writing array to geotiff: %s..."%(filename) + '\n'
    
    saveas = os.path.join(fpath, filename)
    driver = data_meta['driver']
    ncol, nrow = data_meta['ncol'], data_meta['nrow']
    prj = data_meta['prj']
    ext = data_meta['ext']
    n_bands = data_meta['nbands']
    out_raster_ds = driver.Create(saveas, ncol, nrow, n_bands, gdal.GDT_Float64)
    out_raster_ds.SetProjection(prj)
    out_raster_ds.SetGeoTransform(ext)
    
    
    if n_bands > 1:
        for b in range(n_bands):
            out_raster_ds.GetRasterBand(b + 1).WriteArray(data[:, :, b])  
    else:
        out_raster_ds.GetRasterBand(1).WriteArray(data)
	
	# Close dataset
    out_raster_ds = None
    
    #module to clip the results to the study area boundary
def clip_rasters(raster, boundary, data_meta):
    # Must pass name of clip raster file(s), notthe gdal raster object
    clipped_name = raster[:-4]+'_clip.tif'
    clipped = os.path.join(results_dir, clipped_name)
    pixel_res = data_meta['pix_res']

    
    cmd = "gdalwarp.exe -cutline %s -crop_to_cutline -dstnodata 255 -tr %f %f -r bilinear \
    %s %s" %(boundary, float(pixel_res), float(pixel_res), raster, clipped) 
    cmd_info = 'gdalinfo.exe -stats %s'%(clipped)
    subprocess.call(cmd)
    subprocess.call(cmd_info)
    print "%s has been clipped!" %(raster)
    
    
def create_train_test(input_feat, verif_data, w_train_prop, nw_train_prop):

    """
    input_feat = input features that will be used to differentiate between true land classes, \ 
           should be array-like where dimensions correspond to feature categories
      
    verif_data = verification data that is used to create training and testing data
    
    w_train_prop, nw_train_prop = float between 0.0 and 1.0 that represents the proportion of the wetland
        and nonwetland samples that will be used for training, the complement of these variables will \
        be the testing size
    
    """    
    print "Creating training and testing data..." + '\n'
    
    """Training and testing LABEL creation (0s and 1s)"""
    #wetlands = 0 and nonwetlands = 1 in verificaiton dataset
    
    w_all = np.ma.masked_equal(verif_data, 1)
    nw_all = np.ma.masked_equal(verif_data, 0)
    
    #flatten arrays to simplify processing
    w_all_flat = w_all.flatten()
    nw_all_flat = nw_all.flatten()
    
    #get all wetland indices (ie, cannot choose from masked/NaN elements)
    w_indices = np.where(w_all_flat == 0)
    nw_indices = np.where(nw_all_flat == 1)
    true_w_samples = float(np.size(w_indices))
    true_nw_samples = float(np.size(nw_indices))
    
    #TODO: save training/testing stats to file + convert #pixels to area
    
    print "Total number of wetland samples: %d" %(int(true_w_samples)) + '\n'
    print "Total number of nonwetland samples %d" %(int(true_nw_samples)) + '\n'
    
    #get total number of wetland and nonwetland features and calculate number of samples needed
    w_total = len(w_indices[0])
    w_train_n = float(w_train_prop * w_total)
    
    nw_total = len(nw_indices[0])
    nw_train_n = float(nw_train_prop * nw_total)
    
    #choose random indices from wetland and nonwetland arrays to use for training
    #NOTE: np.where returns a list of data, first index is the array we want
    w_rand_indices = np.random.choice(w_indices[0], size = int(w_train_n), replace = False)
    nw_rand_indices = np.random.choice(nw_indices[0], size = int(nw_train_n), replace = False)
    
    #create empty boolean arrays where all elements are False
    w_temp_bool = np.zeros(w_all_flat.shape, bool)
    nw_temp_bool = np.zeros(nw_all_flat.shape, bool)

    #make elements True where the random indices exist
    w_temp_bool[w_rand_indices] = True
    nw_temp_bool[nw_rand_indices] = True

    #training samples are created from true values of the random indices    
    w_train = np.ma.masked_where(w_temp_bool == False, w_all_flat)   
    nw_train = np.ma.masked_where(nw_temp_bool == False, nw_all_flat)
    #testing samples are created from the false values of the random indices (complement)
    w_test = np.ma.masked_where(w_temp_bool == True, w_all_flat)   
    nw_test = np.ma.masked_where(nw_temp_bool == True, nw_all_flat)
    
    #convert masked elements to NaN
    w_train_nans = np.ma.filled(w_train, np.nan)
    nw_train_nans = np.ma.filled(nw_train, np.nan)
    w_test_nans = np.ma.filled(w_test, np.nan)
    nw_test_nans = np.ma.filled(nw_test, np.nan)
    
    #combine training wetlands and nonwetladns into a single array, reshape to write as geotiff
    train_labels = np.where(~np.isnan(nw_train_nans), nw_train_nans, w_train_nans)
    train_labels_2d = train_labels.reshape(verif_data.shape[0], verif_data.shape[1])
    #combine testing wetlands and nonwetlands into a single array, reshape to write as geotiff    
    test_labels = np.where(~np.isnan(nw_test_nans), nw_test_nans, w_test_nans)
    test_labels_2d = np.reshape(test_labels, (verif_data.shape[0], verif_data.shape[1]))
    
       
    """Training and testing FEATURES creation (input variables that are labels either 0 or 1)"""
    #create arrays of input variables within training and testing limits    
    test_features = np.copy(input_feat)
    train_features = np.copy(input_feat)

    test_features[np.isnan(test_labels_2d), : ] = np.nan
    train_features[np.isnan(train_labels_2d), : ] = np.nan
    
    true_ratio = float(true_w_samples / true_nw_samples)
    train_ratio = float(w_train_n / nw_train_n)
    
    print "True wetlands to nonwetlands ratio: %.2f" %(true_ratio) +'\n'
    print "Training wetlands to nonwetlands ratio: %.2f" %(train_ratio) +'\n'
    
    n_train =  float(np.sum(~np.isnan(train_labels)))
    n_test = float(np.sum(~np.isnan(test_labels)))
    train_test_ratio = float(n_train / n_test)

    print "Total training samples to testing samples ratio: %.2f" %(train_test_ratio) +'\n'
    
    return train_labels_2d, test_labels_2d, train_features, test_features


def classify(train_feat, train_labels, test_feat):
    
    """
    -Random Forest classifer is created with global parameters
    -"roi" is region of interest = area to be classified = input variables w/o training areas   
    """

    #Initialize RF model

    print "Initializing Random Forest model with:" 
    print "%d trees" %(n_trees)
    print "%d max tree depth" %(tree_depth)
    print "class weights: W = %d | NW = %d" %(wcw, nwcw) + '\n'
    
    
    rf_clf = RandomForestClassifier( n_estimators = n_trees, max_depth = tree_depth, \
                                    oob_score = True, n_jobs = -1, class_weight = { 0 : wcw, 1 : nwcw } )

   
    
    #train RF model
    #TODO: print shapes w/ nans
    train_X = train_feat[~np.isnan(train_feat[:,:,0]) , :] 
    train_Y = train_labels[~np.isnan(train_labels)] 

    print "Training model..."
    rf_fit = rf_clf.fit(train_X, train_Y)

    #save feature importance    
    
    n_feats = len(rf_fit.feature_importances_)
    importance=[]
    bands = np.arange(1, n_feats+1)
    
    for b, imp in zip(bands, rf_fit.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp.round(2)))
        importance.append('Band {b} importance: {imp} \n'.format(b=b, imp=imp.round(2)))
    
    #execute RF classification and fuzzy classification
	
    print "\n" + "Executing prediction...\n"
    temp_shape = ( test_feat.shape[0] * test_feat.shape[1], test_feat.shape[2])
    test_shaped = test_feat[:, :, :].reshape(temp_shape)
    rf_predict = rf_fit.predict(test_shaped)
    fuzzy_predict = rf_fit.predict_proba(test_shaped)
    print np.shape(fuzzy_predict)
    rf_predict = rf_predict.reshape(test_feat[:, :, 0].shape)
    
    fuzzy_predict_w = fuzzy_predict[:,0].reshape(test_feat[:, :, 0].shape)	
    
    return rf_predict, fuzzy_predict_w, importance

def get_acc(test_labels, pred_vals, importance, fname, results_dir, data_meta, units = None):
    
    #flatten 2d arrays and mask NaNs, must be int for confusion matrix
    t= test_labels.flatten()
    t = t[~np.isnan(t)]
    t = t.astype(int)
    
    p = pred_vals.flatten()
    p = p[~np.isnan(test_labels.flatten())]
    p = p.astype(int)

    
    print "Performing accuracy assessment... \n"
    conf_matrix_pix= confusion_matrix(t, p)
    
    #accuracy
    acc_score= accuracy_score(t, p)

    #specificity (uses confusion matrix)
    TN=conf_matrix_pix[1,1]
    FP=conf_matrix_pix[1,0]
    specificity = float(TN / (TN+FP))
    
    class_report = metrics.classification_report(t, p)

    print class_report
    
    print "Writing results to file... \n"
    #TODO: parse horizontal units from GDAL metadata info and pass to this function
    x_res = data_meta['pix_res']
    y_res = data_meta['pix_res']
    
    #convert confusion matrix to sq km
    conf_matrix_pix= conf_matrix_pix.astype(float)

    if units:
        mult= 1
        units= 'square %s'%units

    else:
        mult=.001**2
        units= 'square km'
    conf_matrix_area=conf_matrix_pix*x_res*y_res*mult


  
    
    #add column and row labels
    confusion_matrix_pd1= pd.DataFrame(conf_matrix_area, index=['actual wetlands (%s)'%(units),'actual nonwetlands (%s)'%(units)], 
                                                                columns=['predicted wetlands (%s)'%(units), 'predicted nonwetlands (%s)'%(units)])
    confusion_matrix_pd= confusion_matrix_pd1.round(3)
    confusion_matrix_pd.loc[u'Σ']= confusion_matrix_pd.sum()
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=0)
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=1)
    
    #convert to series. Use strings to format % sign
    acc_score_pd= pd.Series(acc_score, index=['Accuracy score:'])
    acc_score_pd= acc_score_pd.round(2)
    
    specificity_pd= pd.Series((specificity),
                              index=['Specificity:'])
    specificity_pd=specificity_pd.round(2)
    
    imp_pd=pd.DataFrame(importance)
    imp_pd=imp_pd.round(2)
    
    acc_scores_all= acc_score_pd.append([specificity_pd])
    
    #send unicode classification report output to a pd.DataFrame
    class_report = re.sub(r" +", " ", class_report).replace("avg / total", "avg/total").replace("\n ", "\n")
    class_report_df = pd.read_csv(StringIO("Classes" + class_report), sep=' ', index_col=0) 

    print (class_report_df)
    
    outfile_path = os.path.join(results_dir, fname)
    
    print 'Accuracy report written to: '
    print fname + '\n'
    
    writer=pd.ExcelWriter(outfile_path+ ".xlsx")

    
    confusion_matrix_pd.to_excel(writer, 'Output_accuracy_metrics', startrow=0)
    class_report_df.to_excel(writer, 'Output_accuracy_metrics', header=True, startrow=6)
    acc_scores_all.to_excel(writer, 'Output_accuracy_metrics', header=False, startrow=12)
    imp_pd.to_excel(writer, 'Model_evaluation_metrics', header=False)
    writer.save()
    print 'written to .xlsx file'
    return(conf_matrix_pix, class_report, acc_score, specificity)


def main():
    var_file = "A1_comp_clip.tif"
    var_arr, var_meta = geotiff_to_array(data_dir, var_file)
    var_clean = clean_data(var_arr)

    verif_file = "rte29_vdot_vals_clip_lim.tif"
    verif_arr, verif_meta = geotiff_to_array(data_dir, verif_file)
    verif_clean = clean_data(verif_arr)   
    
    train_labels, test_labels, train_features, test_features = create_train_test(var_clean, verif_clean, w_train_prop, nw_train_prop)
    array_to_geotiff(results_dir,"train_w%.2f_nw%.2f.tif" %(w_train_prop, nw_train_prop), train_labels, verif_meta)
    array_to_geotiff(results_dir,"test_w%.2f_nw%.2f.tif" %(float(1-w_train_prop), float(1-nw_train_prop)) , test_labels, verif_meta)
    array_to_geotiff(results_dir,"train_feats.tif", train_features, var_meta)
    array_to_geotiff(results_dir,"test_feats.tif", test_features, var_meta)    
    rf_predict, fuzzy_predict, importance = classify(train_features, train_labels, var_clean)
    
    rf_results_name = "rf_predict_w%d_nw%d_cws_%d_%d.tif" %(100*w_train_prop, 100*nw_train_prop, wcw, nwcw)
    fuz_results_name = "fuzzy_predict_w%d_nw%d_cws_%d_%d.tif" %(100*w_train_prop, 100*nw_train_prop, wcw, nwcw)
    array_to_geotiff(results_dir,rf_results_name, rf_predict, verif_meta)
    array_to_geotiff(results_dir,fuz_results_name, fuzzy_predict, verif_meta)
    
    clip_rasters(os.path.join(results_dir, rf_results_name), clip_bounds, verif_meta)
    clip_rasters(os.path.join(results_dir, fuz_results_name), clip_bounds, verif_meta)
    
    results_fname = var_file[0:2] + "_w%d_nw%d_cws_%d_%d.xlsx" %(100*w_train_prop, 100*nw_train_prop, wcw, nwcw)   
    conf_matrix_pix, class_report, acc_score, specificity = get_acc(test_labels, rf_predict, importance, results_fname, results_dir, var_meta)

        
if __name__== '__main__':
    time0= time.time()
    main()
    time1=time.time()
    print 'runtime was: {time} seconds'.format(time=(time1-time0))
    sys.exit(0)
    