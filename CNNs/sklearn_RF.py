# -*- coding: utf-8 -*-
"""

"""
import os
from osgeo import gdal
import numpy as np
import sys
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import raster_array_funcspy35 as ra


site = "Site4"
#global variarbles
no_data_flag = -9999
pos_val = 0 #pixel value for the positive class
neg_val = 1 #pixel vlaue for the negative class
pos_prop = 0.5 #percent of positive class to sample for training
neg_prop = 0.5 #percent of negative class to sample for trainig

#set filepaths

outdir = r"D:\3rdStudy_ONeil\Results\RF\{}".format(site)
data_tif = os.path.join(outdir, "{}_comp4.tif".format(site))
#test_labels = r"D:\2ndStudy_ONeil\Tool_testing\results\Site4\Run3\test_w0.850_nw0.920.tif"
test_labels = r"D:\3rdStudy_ONeil\Results\RF\Site4\test_labels_DL_test.tif"
train_labels = r"D:\2ndStudy_ONeil\Tool_testing\results\Site4\Run3\train_w0.150_nw0.080.tif" 
f_path_in = os.path.dirname(data_tif)
f_in = os.path.basename(data_tif)
rf_results_name = "{}_rf_predict.tif".format(site) 


wdir = r"D:\2ndStudy_ONeil\Tool_testing\data\{}\input_variables".format(site)
twi = os.path.join(wdir, "pmSLP50_0.9p_grass_at_twi.tif")
curv = os.path.join(wdir, "pmSLP50_0.9p_curv.tif")
dtw = os.path.join(wdir, "pmSLP50_0.9p_slp_grassrcost_dtw.tif")
ndvi = r"D:\3rdStudy_ONeil\Results\RF\Site4\NVDI.tif"
###flist1 = [twi, curv, dtw, ndvi]
#roi_lim = r"D:\2ndStudy_ONeil\Tool_testing\data\Site2\lims_prj.shp"
#arr, meta = ra.gtiff_to_arr(dtw, "float")
#print(np.min(arr))
#arr_ma = np.ma.masked_invalid(arr)
#print(np.min(arr_ma))
#arr_ma2 = np.ma.masked_values(arr, -9999.)
#print(np.min(arr_ma2))
#arr_out = np.ma.filled(arr_ma, -9999.)
#print(np.min(arr_out))
#arr_tif = ra.arr_to_gtiff(arr, meta, outdir, "dtw.tif")
#sys.exit(0)
##clip = ra.clip_geotif(dtw, outdir, roi_lim)
#sys.exit(0)
#flist=[]
#for f in flist1:
#    clip = ra.clip_geotif(f, outdir, roi_lim)
#    flist.append(clip)
#    
#comp = comp_bands(outdir, flist, "Site1_comp.tif")
#
##clip = ra.clip_geotif(comp, outdir, roi_lim)
#sys.exit(0)


def create_tt_feats(input_feat, train_labels_2d, test_labels_2d, out_dir):
    
    feat_arr, feat_meta = ra.gtiff_to_arr(input_feat, "float")
#    for b in range(4):
#        print(np.min(feat_arr[:, :, b]))
#        
#        band = np.ma.masked_values(feat_arr[:, :, b], -9999.)
#        print(np.min(band))
#       
#        band1 = np.ma.filled(band, np.nan)
#        print(np.min(band1))
#        feat_arr[:, :, b] = band1
        
    train_fname = os.path.join(out_dir, "{}_train_feats.tif".format(site))
    test_fname = os.path.join(out_dir, "{}_test_feats.tif".format(site))
    
    #create arrays of input variables within training and testing limits    
    train_features = np.copy(feat_arr)
    test_features = np.copy(feat_arr)
    print(np.min(train_labels_2d))
    print(np.min(np.ma.masked_invalid(train_labels_2d)))
 
    train_features[np.isnan(train_labels_2d), : ] = -9999.
    test_features[np.isnan(test_labels_2d), : ] = -9999.    
    
    _ = ra.arr_to_gtiff(train_features, feat_meta, out_dir, train_fname)
    _ = ra.arr_to_gtiff(test_features, feat_meta, out_dir, test_fname)
    
    return train_features, test_features

train_labels_2d, meta = ra.gtiff_to_arr(train_labels, dtype="float")
test_labels_2d, meta = ra.gtiff_to_arr(test_labels, dtype="float")

#train_features, test_features = create_tt_feats(data_tif, train_labels_2d, test_labels_2d, outdir)
#train_features, t_meta = ra.gtiff_to_arr(os.path.join(outdir, "{}_train_feats.tif".format(site)), "float")
#test_features, te_meta = ra.gtiff_to_arr(os.path.join(outdir, "{}_test_feats.tif".format(site)), "float")


def classify(train_features, train_labels_2d, test_features, n_trees=300, tree_depth=None, max_f = 'auto'):
    
    train_features1 = np.copy(train_features)
    for b in range(4):
        print(np.min(train_features[:, :, b]))
        
        band = np.ma.masked_invalid(train_features[:, :, b])
        print(np.min(band))
       
        band1 = np.ma.filled(band, -9999.)
        print(np.min(band1))
        train_features1[:, :, b] = band1

#    #train RF model
    train_X = train_features1[~np.ma.masked_values(train_features1[:, :, 0], -9999).mask]
    print(np.min(train_X))
    train_Y = train_labels_2d[~np.ma.masked_values(train_features1[:, :, 0], -9999).mask] 
    print(np.min(train_Y))
    print(np.shape(train_X), np.shape(train_Y))
    #train_X is an array of nonNaN values from train_features, 3D array --> 2D array ([X*Y, Z])
    #train_Y is an array of nonNaN values from train_labels, 2D array --> 1D array (X*Y)
    #both have NO NaN VALUES, which is required for sklearn rf    


    
    #Initialize RF model
    print ("\nInitializing Random Forest model.....\n") 


    rf_clf = RandomForestClassifier( n_estimators = n_trees, max_depth = tree_depth, \
                                    oob_score = True, n_jobs = -1, class_weight = 'balanced',\
                                    max_features = max_f, random_state = 21 )

    print ("Training model...\n")
    rf_fit = rf_clf.fit(train_X, train_Y)    

    #save feature importance    
    n_feats = len(rf_fit.feature_importances_)
    importance=[]
    bands = np.arange(1, n_feats+1)
    
    for b, imp in zip(bands, rf_fit.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp.round(3)))
        importance.append('Band {b} importance: {imp} \n'.format(b=b, imp=imp.round(3)))
    
    #execute RF classification and fuzzy classification
    
    test_feat_masked = np.ma.masked_invalid(test_features)
    test_feat_nonan = np.ma.filled(test_feat_masked, -9999.) #still 3D
    temp_shape = ( test_feat_nonan.shape[0] * test_feat_nonan.shape[1], test_feat_nonan.shape[2] )
    
    test_shaped = test_feat_nonan[:, :, :].reshape(temp_shape)

    print ("Executing prediction...\n")
    rf_predict = rf_fit.predict(test_shaped)
    fuzzy_predict = rf_fit.predict_proba(test_shaped)

    
    #reshape outputs to prepare for geotif export
    rf_predict = rf_predict.reshape(test_features[:, :, 0].shape)    
    fuzzy_predict_pos = fuzzy_predict[:,0].reshape(test_features[:, :, 0].shape)	
    
    return rf_predict, fuzzy_predict_pos, importance

#rf_predict, fuzzy_predict_pos, importance = classify(train_features, train_labels_2d, test_features)
#rf_tif = ra.arr_to_gtiff(rf_predict, meta, outdir, "{}_rf_predict.tif".format(site))
#rf_tif = os.path.join(outdir, "{}_rf_predict.tif".format(site))
rf_tif = r"D:\3rdStudy_ONeil\Results\RF\Site4\rf_predict_DL_test.tif"
rf_predict, _ = ra.gtiff_to_arr(rf_tif, "int")

def get_acc(test_labels_2d, rf_predict, out_dir, pos_val=0, fuzzy_predict_pos=None, importance=None):  
    """ 
    :return: various accuracy metrics
    """ 
    #true labels
    tl_flat = test_labels_2d.ravel()
    tl = tl_flat[~np.isnan(tl_flat)]
   
    #predicted values
    pv_flat = rf_predict.ravel()
    pv = pv_flat[~np.isnan(tl_flat)]
    

    print ("Performing accuracy assessment... \n")
    
    #compute stats
    conf_matrix= metrics.confusion_matrix(tl, pv)
    acc_score= metrics.accuracy_score(tl, pv)
    class_report = metrics.classification_report(tl, pv)
    iou = metrics.jaccard_score(tl, pv, average=None)

    print("Confusion Matrix:\n")
    print(conf_matrix)
    print("----------------------")
    
    print("Accuracy Score:\n")
    print(acc_score)
    print("----------------------")
    
    print("Classification report:\n")
    print(class_report)
    print("----------------------")
    
    print("Jaccard Score (IoU):\n")
    print(iou)
    print("----------------------")

    return conf_matrix, class_report

conf_matrix, class_report = get_acc(test_labels_2d, rf_predict, outdir)



#*******************************run code**************************************#
#
#def run():
#    
#    #open geotiffs
#    array, meta = gtiff_to_arr(f_path_in, f_in)  
#    verif_arr, verif_meta = gtiff_to_arr(os.path.dirname(verification), os.path.basename(verification))
#    
#    #use the verification geotiff to create training and testing files, input percentage from each class to sample for training
#
#    train_labels, test_labels = create_tt_labels(os.path.dirname(verification), os.path.basename(verification), \
#                     pos_prop=pos_prop, neg_prop=neg_prop, pos_val=pos_val, neg_val=neg_val, out_dir=outdir, no_data=no_data_flag)
#    
#    #use train and test labels to index training and testing pixels from data raster
#    train_features, test_features = create_tt_feats(array, meta, train_labels, test_labels, outdir)
#    
#    #execute classification
#    rf_predict, fuzzy_predict_pos, importance = classify(train_features, train_labels, test_features)
#    
#    rf_out = arr_to_gtiff(rf_predict, verif_meta, outdir, rf_results_name, dtype='int')
#    fuz_out = arr_to_gtiff(fuzzy_predict_pos, verif_meta, outdir, fuz_results_name, dtype='int')
#    
#    #get classifier accuracy
#    get_acc(test_labels, rf_predict, fuzzy_predict_pos, importance, outdir, meta, 0)
#    
#    return rf_out, fuz_out
#
#rf_out, fuz_out = run()