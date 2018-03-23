# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 15:08:58 2018

@author: Linnea
"""

#
from sklearn.ensemble import RandomForestClassifier
from osgeo import gdal, gdal_array
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from io import StringIO
import re
import time
import os

results_dir = r"C:\Users\Linnea\Downloads\sklearn_files" 

# composite is a multidimensional raster file where each dimensions represents an input variable (i.e., input variable rasters are "stacked" into one)
composite = "RF_A_clipenvebm.tif"
#train_tif is the training raster file (and path to) used in the RF classification
train_tif = "50p_100ar3_snap.tif"
#known delineations:
true_ds = "p06_vals_str_clip.tif"

composite_input = os.path.join(results_dir,composite)
train_input = os.path.join(results_dir,train_tif)
true_input = os.path.join(results_dir,true_ds)




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
    
    print "Composite's Y size is: "
    print nrow 
    
    print "Composite's X Size is: "
    print ncol
    
    print 'Raster driver: {d}\n'.format(d=driver.ShortName)
    
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
    rf = RandomForestClassifier(n_estimators=50, max_depth=30, oob_score=True, n_jobs = -1) #, class_weight={0:2, 1:100}
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
    print 'predicting...'
    #make a copy to send to accuracy metrics and .xlsx file
    pred=np.zeros(shape=predicted_class.shape)
    np.copyto(pred, predicted_class)
    
    print img.shape
    print img_as_array.shape
    print predicted_class.shape
    predicted_class= predicted_class.reshape(img[:, :, 0].shape)
    print predicted_class.shape
    
    #model evaluation:
    importance=[]
    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp.round(2)))
        importance.append('Band {b} importance: {imp}'.format(b=b, imp=imp.round(2)))

#    save results to a geotiff file so we can view them in ArcGIS
    #deleted all .pyc files that do this, and ensured that .exe files were closed/reopened to fix error
    sklearn_results_name = 'sklearn_RF_geo6.tif'
    out_raster_ds = driver.Create(sklearn_results_name, ncol, nrow, 1, gdal.GDT_Byte)
    out_raster_ds.SetProjection(prj)
    out_raster_ds.SetGeoTransform(ext)
    out_raster_ds.GetRasterBand(1).WriteArray(predicted_class)
    # Close dataset
    out_raster_ds = None

    print 'finished classification'
    
    return(pred,importance)


def clean_data(true_ds,pred): 
    print 'entered clean_data'

    true_gdal = gdal.Open(true_ds, gdal.GA_ReadOnly)
    true_array = true_gdal.ReadAsArray().astype('int')
    
    #make 1 dimensional so that they can be compared by metrics
    pred_array1d= pred.flatten()
    true_array1d= true_array.flatten()
    
    #eliminate NaN pixels (masking)
    pred_arr_vals1 = pred_array1d[pred_array1d != 255]
    true_arr_vals1 = true_array1d[pred_array1d != 255]
    true_arr_vals = true_arr_vals1[true_arr_vals1 != 3]
    pred_arr_vals = pred_arr_vals1[true_arr_vals1 != 3]

    print 'finished clean_data'
    return(true_arr_vals, pred_arr_vals)


def get_acc(true_arr_vals, pred_arr_vals):
    print 'enter get_acc'
    conf_matrix_pix= confusion_matrix(true_arr_vals, pred_arr_vals)
    
    #accuracy
    acc_score= accuracy_score(true_arr_vals, pred_arr_vals)

    #null accuracy
    null= 1-np.mean(true_arr_vals)
    
    #specificity (uses confusion matrix)
    TN=conf_matrix_pix[1,1]
#    TP=conf_matrix_pix[0,0]
#    FN=conf_matrix_pix[0,1]
    FP=conf_matrix_pix[1,0]
    
    specificity = TN / float(TN+FP)
    
    class_report = metrics.classification_report(true_arr_vals, pred_arr_vals)

    print 'finished acc'
    return(conf_matrix_pix,class_report,acc_score,null,specificity)


def write_acc(conf_matrix_pix,class_report,acc_score, null, specificity, importance):
    
    #convert confusion matrix to sq km
    conf_matrix_pix= conf_matrix_pix.astype(float)
    print conf_matrix_pix
    conf_matrix_area=conf_matrix_pix*0.000762
    print type(conf_matrix_area)
    
    
    #add column and row labels
    confusion_matrix_pd1= pd.DataFrame(conf_matrix_area, index=['actual wetlands (square km)','actual non-wetlands (square km)'], 
                                                                columns=['predicted wetlands (sq km)', 'predicted non-wetlands(square km)'])
    confusion_matrix_pd= confusion_matrix_pd1.round(2)
    confusion_matrix_pd.loc[u'Σ']= confusion_matrix_pd.sum()
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=0)
    confusion_matrix_pd[u'Σ'] = confusion_matrix_pd.sum(axis=1)
    
    #convert to series. Use strings to format % sign
    acc_score_pd= pd.Series(acc_score, index=['Accuracy score:'])
    acc_score_pd= acc_score_pd.round(2)
    print acc_score_pd
 #   pd.Series(['{0:.2}%'.format('Accuracy score:')])
    null_pd= pd.Series(null, index=['Null accuracy:'])
    null_pd=null_pd.round(2)
#    null_pd= null_pd.round(2).apply('{:.2%}'.format)
    specificity_pd= pd.Series((specificity.round(2)),
                              index=['Specificity:'])
    specificity_pd=specificity_pd.round(2)
 #   specificity_pd[0]= specificity_pd.round(2).apply('{:.2%}'.format)
    imp_pd=pd.DataFrame(importance)
    imp_pd=imp_pd.round(2)
    
    acc_scores_all= acc_score_pd.append([specificity_pd, null_pd])
    print acc_scores_all
    
    #send unicode classification report output to a pd.DataFrame
    class_report = re.sub(r" +", " ", class_report).replace("avg / total", "avg/total").replace("\n ", "\n")
    class_report_df = pd.read_csv(StringIO("Classes" + class_report), sep=' ', index_col=0)        
    print (class_report_df)
    

    
    print 'writing to files'
    writer=pd.ExcelWriter('C:\Users\Linnea\Documents\GitHubWetlands\wetland_identification\Post_Classification\sklearn_acc_metrics3.xlsx')
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
    print 'written to .xlsx file'
    return()


def main(train_tif,composite, true_ds):

    pred_class, f_imp= classify(composite, train_tif,true_ds)
    pred_clean, true_clean=clean_data(true_ds,pred_class)
    a,b,c,d,e= get_acc(pred_clean, true_clean)
    write_acc(a,b,c,d,e,f_imp)
    

if __name__== '__main__':
    time0= time.time()
    main(train_input,composite_input, true_input)
    time1=time.time()
    print 'runtime was: {time} seconds'.format(time=(time1-time0))

