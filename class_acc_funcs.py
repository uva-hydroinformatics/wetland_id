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
from sklearn.metrics import *
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
import wetland_id_defaults as default
from raster_array_funcs import *
import matplotlib.pyplot as plt


    
def create_tt_labels(verif_tif, w_train_prop, nw_train_prop, out_dir):

    verif_arr, verif_meta = geotif_to_array(verif_tif)

    verif_int = verif_arr.astype(int)   #make sure there are 2 unqiue values - wetland (0) and nonwetland (1) 

    verif_data = clean_array(verif_int)
    
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
    w_all = np.ma.masked_values(verif_data, 1.)
    nw_all = np.ma.masked_values(verif_data, 0.)
    print np.unique(w_all), np.unique(nw_all)
    #flatten arrays to simplify processing
    w_all_flat = w_all.flatten()
    nw_all_flat = nw_all.flatten()
    print np.unique(w_all_flat), np.unique(nw_all_flat)
    #get all wetland indices (ie, cannot choose from masked/NaN elements)
    w_indices = np.where(w_all_flat == 0)
    nw_indices = np.where(nw_all_flat == 1)
    
    #get total number of wetland and nonwetland features and calculate number of samples needed
    w_total = float(len(w_indices[0]))
    w_train_n = float(w_train_prop * w_total)
    
    nw_total = float(len(nw_indices[0]))
    nw_train_n = float(nw_train_prop * nw_total)
   
    print "Total number of verification wetland samples: %d" %(int(w_total)) + '\n'
    print "Total number of verification nonwetland samples %d" %(int(nw_total)) + '\n'
    
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
    
    w_train_flt, nw_train_flt = w_train.astype(float), nw_train.astype(float)
    w_test_flt, nw_test_flt = w_test.astype(float), nw_test.astype(float)
    
    #convert masked elements to NaN
    w_train_nans = np.ma.filled(w_train_flt, np.nan)
    nw_train_nans = np.ma.filled(nw_train_flt, np.nan)
    w_test_nans = np.ma.filled(w_test_flt, np.nan)
    nw_test_nans = np.ma.filled(nw_test_flt, np.nan)
    
    #combine training wetlands and nonwetladns into a single array, reshape to write as geotiff
    train_labels = np.where(~np.isnan(nw_train_nans), nw_train_nans, w_train_nans)
    
    #combine testing wetlands and nonwetlands into a single array, reshape to write as geotiff    
    test_labels = np.where(~np.isnan(nw_test_nans), nw_test_nans, w_test_nans)

    #stats
    true_ratio = float(w_total / nw_total)
    train_ratio = float(w_train_n / nw_train_n)
    
    print "True wetlands to nonwetlands ratio: %.3f" %(true_ratio) +'\n'
    print "Training wetlands to nonwetlands ratio: %.3f" %(train_ratio) +'\n'
    
    train_labels_2d = np.reshape(train_labels, (verif_data.shape[0], verif_data.shape[1]))
    test_labels_2d = np.reshape(test_labels, (verif_data.shape[0], verif_data.shape[1]))

    train_labels_tif = array_to_geotif(train_labels_2d, verif_meta, out_dir, "train_w%.3f_nw%.3f.tif" %(w_train_prop, nw_train_prop))
    test_labels_tif = array_to_geotif(test_labels_2d, verif_meta, out_dir,"test_w%.3f_nw%.3f.tif" %(float(1-w_train_prop), float(1-nw_train_prop)))

    return train_labels_2d, test_labels_2d

def create_tt_feats(input_feat, train_labels_2d, test_labels_2d, out_dir):
    
    feat_arr, feat_meta = geotif_to_array(input_feat)
    feat_clean = clean_array(feat_arr)
    feat_arr = np.ma.filled(feat_clean, np.nan)
    
    out_base = input_feat.split('\\')[-1]
    train_fname = out_base[:-4] + "train_feats.tif"
    test_fname = out_base[:-4] + "test_feats.tif"
    """Training and testing FEATURES creation (input variables that are labeled either 0 or 1)"""
    #create arrays of input variables within training and testing limits    
    train_features = np.copy(feat_arr)
    test_features = np.copy(feat_arr)
    
    train_features[np.isnan(train_labels_2d), : ] = np.nan
    test_features[np.isnan(test_labels_2d), : ] = np.nan    
    
    train_feat_tif = array_to_geotif(train_features,feat_meta, out_dir, train_fname)
    test_feat_tif = array_to_geotif(test_features, feat_meta, out_dir, test_fname)
    
    return train_features, test_features

def classify(train_features, train_labels_2d, test_features, n_trees, tree_depth, wcw, nwcw):

    #Initialize RF model

    print "Initializing Random Forest model with:" 
    print "%d trees" %(n_trees)
    print "%s max tree depth" %str(tree_depth)
    print "class weights: W = %s | NW = %s" %(str(wcw), str(nwcw)) + '\n'
    
    if type(wcw) is int:
        rf_clf = RandomForestClassifier( n_estimators = n_trees, max_depth = tree_depth, \
                                        oob_score = True, n_jobs = default.n_proc, class_weight = { '0' : wcw, '1' : nwcw },\
                                        max_features = 'auto', random_state = 21 )
    else:
        rf_clf = RandomForestClassifier( n_estimators = n_trees, max_depth = tree_depth, \
                                        oob_score = True, n_jobs = default.n_proc, class_weight = 'balanced',\
                                        max_features = 'auto', random_state = 21 )
    #train RF model
    train_X = train_features[~np.isnan(train_features[:, :, 0]), :]

    train_Y = train_labels_2d[~np.isnan(train_features[:, :, 0])] 
#    train_Y = train_labels_2d[~np.isnan(train_labels_2d)] 

    train_Y1 = train_Y.astype(int)
    train_Y2 = train_Y1.astype(str)

    #train_X is an array of nonNaN values from train_features, 3D array --> 2D array ([X*Y, Z])
    #train_Y is an array of nonNaN values from train_labels, 2D array --> 1D array (X*Y)
    #both have NO NaN VALUES, which is required for sklearn rf    

    print "Training model..."
    rf_fit = rf_clf.fit(train_X, train_Y2)    

    #save feature importance    
    
    n_feats = len(rf_fit.feature_importances_)
    importance=[]
    bands = np.arange(1, n_feats+1)
    
    for b, imp in zip(bands, rf_fit.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp.round(3)))
        importance.append('Band {b} importance: {imp} \n'.format(b=b, imp=imp.round(3)))
    
    #execute RF classification and fuzzy classification


    #cannot delete NaN values of test_features, because we will not be able to export to geotiff with extents of original ROI
    #convert test_features to 2D and fill nan values with other # (-9999)
    
    test_feat_masked = np.ma.masked_invalid(test_features)
    test_feat_nonan = np.ma.filled(test_feat_masked, -9999.) #still 3D
    temp_shape = ( test_feat_nonan.shape[0] * test_feat_nonan.shape[1], test_feat_nonan.shape[2] )
    
    test_shaped = test_feat_nonan[:, :, :].reshape(temp_shape)

    #the array used in rf.predict() and rf.predict_proba() MUST be the same size as the original ROI, eventhough those extents include 
    #many NAN values. Here, we assign a dummy value (-9999) to NaN locations to maintain shape. There is no training data for these
    #values, so the accuracy for this class should be zero but not affect the accuracy of W/NW classes

    print "\n" + "Executing prediction...\n"

    
    rf_predict = rf_fit.predict(test_shaped)
    fuzzy_predict = rf_fit.predict_proba(test_shaped)

    
    #reshape outputs to prepare for geotif export
    rf_predict = rf_predict.reshape(test_features[:, :, 0].shape)
    
    fuzzy_predict_w = fuzzy_predict[:,0].reshape(test_features[:, :, 0].shape)	
    
    
    return rf_predict, fuzzy_predict_w, importance

def get_acc(test_labels_2d, rf_predict, fuzzy_predict_w, importance, fname, results_dir, data_meta, units = None):
    
    #flatten 2d arrays and mask NaN, must be int for confusion matrix
    #test_labels_2D already has nan values...so these are masked
    tl_flat = test_labels_2d.flatten()
    tl_nonans = tl_flat[~np.isnan(tl_flat)]
    tl = tl_nonans.astype(int) #true labels

    pv_flat = rf_predict.flatten()
    pv_nonans = pv_flat[~np.isnan(tl_flat)]
    pv = pv_nonans.astype(int) #predicted values

    fuzz_flat = fuzzy_predict_w.flatten()
    fuzz_nonans = fuzz_flat[~np.isnan(tl_flat)] #fuzzy predictions for wetland class (=0)

    
    print "Performing accuracy assessment... \n"
    conf_matrix_pix= confusion_matrix(tl, pv)
    
    #accuracy
    acc_score= accuracy_score(tl, pv)

    #specificity
    specificity = recall_score(tl, pv, pos_label = 1) #use recall to get specificity, pos lable here is a lie
    
    #fbeta score
    fbeta = fbeta_score(tl, pv, beta = default.beta, pos_label = 0)
    
    #Compute AU Precision Recall Curve (average precision by sklearn)
     #do some funky stuff to make sklearn recognize positive class
    tl_temp = np.copy(tl)
    tl_temp[tl_temp == 1] = 0 #switch 0s and 1s because sklearn average precision method defaults positive class to 1
    tl_temp[tl == 0] = 1
    ap_score = average_precision_score(tl_temp, fuzz_nonans, average='weighted')
    precision, recall, thresholds = precision_recall_curve(tl, fuzz_nonans, pos_label = 0)
    auc_pr = auc(recall, precision)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap_score))
    prc_out = 'prec_rec_curve.png'
    plt.savefig(prc_out, dpi = 300)
    print "Average Precision Recall score (or Area Under Precision Recall Curve): %f \n" %(ap_score)
    
    #Get ROC
    fpr, tpr, thresholds = roc_curve(tl, fuzz_nonans, pos_label = 0)
    auroc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % auroc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    roc_out = 'roc.png'
    plt.savefig(roc_out, dpi = 300)
    plt.show()
    
    class_report = metrics.classification_report(tl, pv)

    print class_report
    print " Accuracy Score: %f \n Specificity: %f \n F%.1f Score: %f\n AP Score: %f \n" \
            %(acc_score, specificity, default.beta, fbeta, ap_score)
    
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
    confusion_matrix_pd= confusion_matrix_pd1.round(4)
    confusion_matrix_pd.loc[u'Σ']= confusion_matrix_pd.sum()
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=0)
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=1)
    
    #convert to series. Use strings to format % sign
    acc_score_pd = pd.Series(acc_score, index=['Accuracy score:'])
    acc_score_pd = acc_score_pd.round(4)
    
    #add specificity
    specificity_pd = pd.Series(specificity, index=['Specificity:'])
    specificity_pd = specificity_pd.round(4)
    
    #add F-beta score
    fbeta_pd = pd.Series(fbeta, index=['F%.1f Score:' %(default.beta)])
    fbeta_pd = fbeta_pd.round(4)
    
    #add Average Precision (~Area under Precision Recall Curve)
    ap_pd = pd.Series(ap_score, index=['Average Precision Score:'])
    ap_pd = ap_pd.round(4)
    
    #add Area Under ROC
    auroc_pd = pd.Series(auroc, index=['Area Under ROC:'])
    auroc_pd = auroc_pd.round(4)
    
    imp_pd=pd.DataFrame(importance)
    imp_pd=imp_pd.round(4)
    
    acc_scores_all= acc_score_pd.append([specificity_pd, fbeta_pd, ap_pd, auroc_pd])
    
    #send unicode classification report output to a pd.DataFrame
    class_report = re.sub(r" +", " ", class_report).replace("avg / total", "avg/total").replace("\n ", "\n")
    class_report_df = pd.read_csv(StringIO("Classes" + class_report), sep=' ', index_col=0) 
    
    p_r_df = pd.DataFrame({'precision': precision, 'recall': recall})
    roc_df = pd.DataFrame({'TPR': tpr, 'FPR': fpr})
    
    
    outfile_path = os.path.join(results_dir, fname)
    
    print 'Accuracy report written to: '
    print fname + '\n'
    
    writer=pd.ExcelWriter(outfile_path)

    confusion_matrix_pd.to_excel(writer, sheet_name = 'Output_accuracy_metrics', startrow=0)
    class_report_df.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=True, startrow=6)
    acc_scores_all.to_excel(writer, sheet_name = 'Output_accuracy_metrics', header=False, startrow=12)
    imp_pd.to_excel(writer, sheet_name = 'Variable_Imp', header=False)
    p_r_df.to_excel(writer, sheet_name = 'Performance_Curves', header=True, startrow=0)
    roc_df.to_excel(writer, sheet_name = 'Performance_Curves', header=True, startrow=0, startcol = 3)
    
    worksheet = writer.sheets['Performance_Curves']
    worksheet.insert_image('H1', prc_out)
    worksheet.insert_image('H21', roc_out)
    
    writer.save()
    print 'written to %s' %(fname)
    
    return fname


def main(comp, verif_tif, results_dir = default.roi_results, n_trees = default.n_trees, \
         tree_depth = default.tree_depth, wcw = default.wcw, nwcw = default.nwcw,\
         w_train_prop = default.w_train_prop, nw_train_prop = default.nw_train_prop):
        
    verif_arr, verif_meta = geotif_to_array(verif_tif)
    nans = np.where(np.isnan(verif_arr))

    train_labels_2d, test_labels_2d = create_tt_labels(verif_tif, w_train_prop, nw_train_prop, results_dir)
    
    var_arr, var_meta = geotif_to_array(comp)

    train_features, test_features = create_tt_feats(comp, train_labels_2d, test_labels_2d, results_dir)

    rf_predict, fuzzy_predict_w, importance = classify(train_features, train_labels_2d,\
                                                       test_features, n_trees, tree_depth, wcw, nwcw)

    rf_predict[nans] = np.nan
    fuzzy_predict_w[nans] = np.nan

    comp_base = comp.split('\\')[-1]

    rf_results_name = "%s_rf_predict_w%.3f_nw%.3f_cws_%s_%s.tif" \
    %(comp_base[:-4], 100*w_train_prop, 100*nw_train_prop, str(wcw), str(nwcw))
    
    fuz_results_name = "%s_fuzzy_predict_w%.3f_nw%.3f_cws_%s_%s.tif" \
    %(comp_base[:-4], 100*w_train_prop, 100*nw_train_prop, str(wcw), str(nwcw))

    rf_out = array_to_geotif(rf_predict, verif_meta, results_dir, rf_results_name)
    fuz_out = array_to_geotif(fuzzy_predict_w, verif_meta, results_dir, fuz_results_name)

    results_fname = os.path.join(results_dir, "%s_w%.3f_nw%.3f_cws_%s_%s.xlsx" \
                                 %(comp_base[:-4], 100*w_train_prop, 100*nw_train_prop, str(wcw), str(nwcw)))   
    results = get_acc(test_labels_2d, rf_predict, fuzzy_predict_w, importance, results_fname, results_dir, var_meta)
    
    return rf_out, fuz_out, results
        
if __name__== '__main__':
    time0= time.time()
    main()
    time1=time.time()
    print 'runtime was: {time} seconds'.format(time=(time1-time0))
    sys.exit(0)
    