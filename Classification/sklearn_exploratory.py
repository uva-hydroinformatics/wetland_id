# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 17:07:19 2018

@author: Linnea
"""

import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from osgeo import gdal, gdal_array, ogr
import numpy as np
import sys
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, confusion_matrix, classification_report, precision_recall_curve, roc_auc_score
import pandas as pd
from io import StringIO
import re
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score

# composite is a multidimensional raster file where each dimensions represents an input variable (i.e., input variable rasters are "stacked" into one)
composite = r"C:\Users\Linnea\Downloads\sklearn_files\RF_A_clipenvebm.tif"

#train_tif is the training raster file (and path to) used in the RF classification
train_tif = r"C:\Users\Linnea\Downloads\sklearn_files\50p_100ar3_snap.tif"

#test is the VDOT wetland delineations raster file, training data has not been separated from these delineations
true_ds = r"C:\Users\Linnea\Downloads\sklearn_files\p06_vals_str_clip.tif"

def classify(composite,train_tif, true_ds):

    comp_ds = gdal.Open(composite, gdal.GA_ReadOnly)
    
    train_ds = gdal.Open(train_tif, gdal.GA_ReadOnly)
    
    test_ds = gdal.Open(true_ds, gdal.GA_ReadOnly)
    
    num_bands = comp_ds.RasterCount
    
    print "Number of bands in composite: {n}\n".format(n=num_bands) #bands = dimensions
    
    #need these attributes to write a new georeferenced file
    driver = comp_ds.GetDriver()
    prj = comp_ds.GetProjection()
    ncol = comp_ds.RasterXSize
    nrow = comp_ds.RasterYSize
    ext = comp_ds.GetGeoTransform()
    
    gdal.UseExceptions()
    gdal.AllRegister()
    
    num_bands = train_ds.RasterCount
    print "Number of bands in training raster: {n}\n".format(n=num_bands)
    
    train_arr = train_ds.GetRasterBand(1).ReadAsArray().astype('int')
    
    test_arr = test_ds.GetRasterBand(1).ReadAsArray().astype('int')
    
    print "Shape of entire roi training area: {n} \n".format(n=train_arr.shape)
    
    #printing to quickly find the NaN of train_arr, \
    #data values should be either =0 for wetland or =1 for non-wetland another number must be NaN
    #this part needs to be automated in the future
    print 'train_arr min,max, shape: '
    print np.min(train_arr)
    print np.max(train_arr)
    print train_arr.shape
    print 'test_arr min, max, shape:'
    print np.min(test_arr)
    print np.max(test_arr)
    print test_arr.shape
    
    #creating an empty array that is the size I want
    img = np.zeros((comp_ds.RasterYSize, comp_ds.RasterXSize, comp_ds.RasterCount), \
                   gdal_array.GDALTypeCodeToNumericTypeCode(comp_ds.GetRasterBand(1).DataType))
    print 'The shape of img is:'
    print img.shape
    #populating the empty array
    for b in range(img.shape[2]):
        img[:, :, b] = comp_ds.GetRasterBand(b + 1).ReadAsArray()
    
    #masking img array to only include where train_arr is not NaN (i.e., <3, the NaN value)
    #this is not a sophisticated way to do it
        
    X = img[train_arr < 3, :] #pixels that will be used for training
    Y = train_arr[train_arr < 3] #"class labels" that indicate which class (w/nw/) training pixels belong to
    
    print('Our X matrix is sized: {sz}'.format(sz=X.shape))
    print('Our y array is sized: {sz}'.format(sz=Y.shape))
    
    
    # Initialize our model with 100 trees
    rf = RandomForestClassifier(n_estimators=50, max_depth=30, oob_score=True, n_jobs = -1, class_weight={0:100, 1:2}) #
    print 'model initialized...now fitting'
    
    # Fit our model to training data. Use class weights not sample weights (sw is to give more reliable data an edge)
    rf = rf.fit(X, Y)
    print 'fitted...now oob_score'
    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
    bands = [1, 2, 3]
 
    #need to flatten array into 2D to classify   
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    
    img_as_array = img[:, :, :].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape,
                                            n=img_as_array.shape))
    
    # Now predict for each pixel
    predicted_class= rf.predict(img_as_array)
    pred=np.zeros(shape=predicted_class.shape)
    np.copyto(pred, predicted_class)
#    pred=np.copy(predicted_class)
#    pred_precision_recall=np.copy(predicted_class)
    
    print img.shape
    print img_as_array.shape
    print predicted_class.shape
    predicted_class= predicted_class.reshape(img[:, :, 0].shape)
    print predicted_class.shape
    
    #########model evaluation
        #show variable importance output from RF classification
#    for b, imp in zip(bands, rf.feature_importances_):
#        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
        
    importance=[]
    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
        importance.append('Band {b} importance: {imp}'.format(b=b, imp=imp))
        
    
    #pred_true_ds= (rf.predict_proba(img_as_array)[:,0]>.4).astype(int) #should return same as predict
    pred_true_ds0= rf.predict_proba(img_as_array)
    
    proba_df0= pd.DataFrame(pred_true_ds0, dtype=float)
    proba_df0.columns=['wetlands','non-wetlands']
    print 'prbab_df head, mean:'
    print proba_df0.head
    print proba_df0.mean
    plt.plot(proba_df0['wetlands'])
    
    pred_true_list=[]
    for i in pred_true_ds0[:,0]:
        if i>0.02:
            pred_true_list.append(0)
        else:
            pred_true_list.append(1)
    pred_true_ds= np.asarray(pred_true_list)
#    proba_df= pd.DataFrame(pred_true_ds, dtype=float)
#    print 'prbab_df head, mean:'
#    print proba_df.head
#    print proba_df.mean
#    plt.plot(proba_df[:,1])

    plt.show()
    
    print pred_true_ds[0]
#        pred_true_ds= (rf.predict_proba(img_as_array)[:,1]>.5).astype(int) #should return same as predict
#    print pred_true_ds[:,:]
    
    print 'pred_true_ds shape: '
    print pred_true_ds.shape

    test_arr1d= test_arr.flatten()
    print test_arr1d.shape
#    pred_true_ds1=pred_true_ds[:,0]
#    print pred_true_ds1.shape
#    print pred_true_ds1[:,0]
    
    
    precision, recall, thresholds = precision_recall_curve(test_arr1d[test_arr1d==0], pred_true_ds[test_arr1d==0])
    average_precision= average_precision_score(test_arr1d[test_arr1d==0], pred_true_ds[test_arr1d==0], average= 'micro')
#    average_precision= average_precision_score(pred_true_ds[:,0], img_as_array, average= 'micro')
    
#    print 'av precision for pos values is %3.2f'%average_precision
    
    plt.step(recall, precision, color='b', alpha=0.2,
              where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                      color='b')
     
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    return(pred,importance, pred_true_ds)
    
def clean_data(true_ds,pred): 
    print 'entered clean_data'
    
    ################################## error here:
    
#    pred_gdal = gdal.Open(pred, gdal.GA_ReadOnly)
    true_gdal = gdal.Open(true_ds, gdal.GA_ReadOnly)
    
    print 'opened as gdal'
    #create predicted and true arrays from images
#    pred_array = pred_gdal.ReadAsArray().astype('int')
    
    true_array = true_gdal.ReadAsArray().astype('int')
    #change to pred
    print type(pred)
    
    print np.shape(pred)
    print np.shape(true_array)
    
    #make 1 dimensional so that they can be compared by metrics
    pred_array1d= pred.flatten()
    
    true_array1d= true_array.flatten()
    
    #eliminate NaN pixels (masking)
    pred_arr_vals1 = pred_array1d[pred_array1d != 255]
    true_arr_vals1 = true_array1d[pred_array1d != 255]
    true_arr_vals = true_arr_vals1[true_arr_vals1 != 3]
    pred_arr_vals = pred_arr_vals1[true_arr_vals1 != 3]
    
    
    print pred_arr_vals
    print true_arr_vals
    print 'finished clean_data'
    return(true_arr_vals, pred_arr_vals)


def get_acc(true_arr_vals, pred_arr_vals, pred_true_ds):
    print 'entered get_acc'
    conf_matrix_pix= confusion_matrix(true_arr_vals, pred_arr_vals)
    print true_arr_vals.shape
    print pred_arr_vals.shape
    print pred_true_ds.shape
    
    #accuracy score, null accuracy, specificity
    # =============================================================================
    #accuracy
    acc_score= accuracy_score(true_arr_vals, pred_arr_vals)
    #pred_true_ds1= pred_true_ds[:,0]
    pred_true_ds1=pred_true_ds[true_arr_vals]
    print pred_true_ds1
    print pred_true_ds1.shape
    
    pp_report = metrics.classification_report(true_arr_vals, pred_true_ds[true_arr_vals])
    print pp_report
    #null accuracy
    null= 1-np.mean(true_arr_vals)
    
    
    #specificity (uses confusion matrix)
    TN=conf_matrix_pix[1,1]
#    TP=conf_matrix_pix[0,0]
#    FN=conf_matrix_pix[0,1]
    FP=conf_matrix_pix[1,0]
    
    specificity = TN / float(TN+FP)
    
    class_report = metrics.classification_report(true_arr_vals, pred_arr_vals)
    
    #classification report (precision= tp/tp+fp; recall=tp/(tp+fn); 
    #f1= mean of precision and recall(recall weighted by a factor of 1)); support= #y-true in each class
    # =============================================================================
    #function to format unicode into a dataframe

    
    
    
    #precision (TP/total pred) -recall (tot pred P/tot actual P curve: report for 0, not 1
    # =============================================================================
    

    print 'finished acc'
    return(conf_matrix_pix,class_report,acc_score,null,specificity, pp_report)
#write everything to an excel file
# =============================================================================

def write_acc(conf_matrix_pix,class_report,acc_score, null, specificity, importance,proba_pred):
    #imp = class.var_importance
    
    #convert to sq km
    conf_matrix_pix= conf_matrix_pix.astype(float)
    print conf_matrix_pix
    conf_matrix_area=conf_matrix_pix*0.000762
    print type(conf_matrix_area)
    
    
    #add column and row labels
    confusion_matrix_pd= pd.DataFrame(conf_matrix_area, index=['actual wetlands (square km)', 'actual non-wetlands (square km)'], columns=['predicted wetlands (sq km)', 'predicted non-wetlands(square km)'])
    confusion_matrix_pd.loc[u'Σ']= confusion_matrix_pd.sum()
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=0)
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=1)
    
    acc_score_pd= pd.Series(acc_score, index=['Fraction of correctly predicted points:'])
    null_pd= pd.Series(null, index=['Null accuracy:'])
    specificity_pd= pd.Series(specificity,
                              index=['Specificity:'])
    
    print 'write prob_pred to a file'
        
    imp_pd=pd.DataFrame(importance)
    
    
#    def report_to_df(class_report):
#        class_report = re.sub(r" +", " ", class_report).replace("avg / total", "avg/total").replace("\n ", "\n")
#        report_df = pd.read_csv(StringIO("Classes" + class_report), sep=' ', index_col=0)        
#        return(report_df)
#    
#    
#    class_report_df = report_to_df(class_report)
    #send unicode report output to a pd.DataFrame
    class_report = re.sub(r" +", " ", class_report).replace("avg / total", "avg/total").replace("\n ", "\n")
    class_report_df = pd.read_csv(StringIO("Classes" + class_report), sep=' ', index_col=0)        
    
    print (class_report_df)
    
    acc_scores_all= acc_score_pd.append([specificity_pd, null_pd])
    print acc_scores_all
    
    print 'writing to files'
    writer=pd.ExcelWriter('C:\Users\Linnea\Documents\GitHubWetlands\wetland_identification\Post_Classification\sklearn_acc_metrics_cw_98-2.xlsx')
#    header_format = 'Output_accuracy_metrics'.add_format({
#    'bold': True,
#    'text_wrap': True,
#    'valign': 'top',
#    'fg_color': '#D7E4BC',
#    'border': 1})
    
    confusion_matrix_pd.to_excel(writer, 'Output_accuracy_metrics', startrow=0)
    class_report_df.to_excel(writer, 'Output_accuracy_metrics', header=True, startrow=6)
    acc_scores_all.to_excel(writer, 'Output_accuracy_metrics', header=False, startrow=12)
    imp_pd.to_excel(writer, 'Model_evaluation_metrics', header=False)
    writer.save()
    return()


def main(train_tif,composite, true_ds):
    #do it all
    #get accuracy results from the accuracy module
    print 'in the main function'
    pred_class, f_imp, proba_pred = classify(composite, train_tif,true_ds)
    #true_ds is global, pred_class is from classify()
    pred_clean, true_clean=clean_data(true_ds,pred_class)
    a,b,c,d,e,pp = get_acc(pred_clean, true_clean, proba_pred)
    write_acc(a,b,c,d,e,f_imp,pp)

main(train_tif,composite, true_ds)
