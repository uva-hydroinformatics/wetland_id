# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:59:31 2018

@author: Linnea
"""

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
import pandas as pd
from io import StringIO
import re
import time
import os
import ntpath


results_dir = r"C:\Users\Linnea\Documents\sklearn_files3-8\sklearn_files" 

## composite is a multidimensional raster file where each dimensions represents an input variable (i.e., input variable rasters are "stacked" into one)
composite = "A1_comp_clip.tif"

##known delineations:
true_ds = "rte29_vdot_vals_clip_lim.tif"


composite_input = os.path.join(results_dir,composite)

true_input = os.path.join(results_dir,true_ds)


#units='feet'
def tt_split(composite, true_ds):
    comp_ds = gdal.Open(composite, gdal.GA_ReadOnly)
    
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
    
    test_arr = test_ds.GetRasterBand(1).ReadAsArray().astype('int')
    print test_arr.shape
    
    img = np.zeros((comp_ds.RasterYSize, comp_ds.RasterXSize, comp_ds.RasterCount), \
                   gdal_array.GDALTypeCodeToNumericTypeCode(comp_ds.GetRasterBand(1).DataType))
    print 'The shape of img is:'
    print img.shape
    
#    populating the empty array
    for b in range(img.shape[2]):
        img[:, :, b] = comp_ds.GetRasterBand(b + 1).ReadAsArray()
    
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    
    img_as_array = img[:, :, :].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=img.shape,
                                            n=img_as_array.shape))
    
    new_shape_test= (test_arr.shape[0]*test_arr.shape[1])
    
    test_shaped=test_arr[:,:].reshape(new_shape_test)
    print test_shaped.shape

    
    X_train, X_test, y_train, y_test = train_test_split(img_as_array,
                                                        test_shaped,
                                                        train_size=0.1)
    
    print 'Y_train 0, 1:'
    print np.size(y_train[y_train==0])
    print np.size(y_train[y_train==1])
    
    print type(X_train)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape
    return (X_train, X_test, y_train, y_test, img)


def classify(X_train, X_test, y_train, img):
    
    # Initialize our model with 100 trees
    rf = RandomForestClassifier(n_estimators=50, max_depth=30, oob_score=True, n_jobs = -1) #class_weight={0:1, 1:.02}
    print 'model initialized...now fitting'
    
    # Fit our model to training data. Use class weights not sample weights (sw is to give more reliable data an edge)
    rf = rf.fit(X_train, y_train)
    print 'fitted...now oob_score'
    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
    bands = [1, 2, 3]
 
#    #need to flatten array into 2D to classify   
#    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
#    
#    img_as_array = img[:, :, :].reshape(new_shape)
#    print('Reshaped from {o} to {n}'.format(o=img.shape,
#                                            n=img_as_array.shape))
    
    # Now predict for each pixel
    predicted_class= rf.predict(X_test)
    print 'predicting...'


    
    #make a copy of 1d array to send to accuracy metrics and .xlsx file
    pred=np.zeros(shape=predicted_class.shape)
    np.copyto(pred, predicted_class)
    
    #img is a different shape here than in previous script, 
    #I haven't figured out what shape predicted_class should be to print to geotiff -LS 3/16/18
#    print predicted_class.shape
#    predicted_class= predicted_class.reshape(img[:, :, 0].shape)
#    print predicted_class.shape

    
#    print 'mean_abs_error is: '
 #   print metrics.mean_absolute_error(test_arr, predicted_class)
    
    
    
    #model evaluation:
    importance=[]
    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp.round(2)))
        importance.append('Band {b} importance: {imp}'.format(b=b, imp=imp.round(2)))

    
    
    #save results to a geotiff file so we can view them in ArcGIS
    #deleted all .pyc files that do this, and ensured that python .exe files were closed/reopened to fix error
#    if composite==None:
#        file_name='sklearn_accuracy_metrics'
#    else:
#        file_name=ntpath.basename(composite).split('.')[0]
#        
#    sklearn_results_name = 'sklearn_rf_{}'.format(ntpath.basename(composite))
#    print 'results geotiff is called: '
#    print sklearn_results_name
#    out_raster_ds = driver.Create(sklearn_results_name, ncol, nrow, 1, gdal.GDT_Byte)
#    out_raster_ds.SetProjection(prj)
#    out_raster_ds.SetGeoTransform(ext)
#    out_raster_ds.GetRasterBand(1).WriteArray(predicted_class)
#    # Close dataset
#    out_raster_ds = None

    print 'finished classification'
    
    return(pred,importance)


def clean_data(y_test,pred): 
    print 'entered clean_data'
    
    #make 1 dimensional so that they can be compared by metrics
    pred_array1d= pred.flatten()
#    y_test= y_test.flatten()


    
    #eliminate NaN pixels (masking)
    pred_arr_vals = pred_array1d[pred_array1d != 3]
    true_arr_vals = y_test[pred_array1d != 3]
#    true_arr_vals = true_arr_vals1[true_arr_vals1 != 3]
#    pred_arr_vals = pred_arr_vals1[true_arr_vals1 != 3]

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


def write_acc(conf_matrix_pix,class_report,acc_score, null, specificity, importance, composite, units=None):
    
    
    comp_ds = gdal.Open(composite, gdal.GA_ReadOnly)
    x_res=comp_ds.GetGeoTransform()[1]
    print "comp's x pixel size (w-e) is: "
    print x_res
    
    y_res= -comp_ds.GetGeoTransform()[5]
    print "comp's y (s-n) pixel size is: "
    print y_res
    
    #convert confusion matrix to sq km
    conf_matrix_pix= conf_matrix_pix.astype(float)
    print conf_matrix_pix
    if units:
        mult= 1
        units= 'square %s'%units

    else:
        mult=.001**2
        units= 'square km'
    conf_matrix_area=conf_matrix_pix*x_res*y_res*mult
    print type(conf_matrix_area)
    
    
    #add column and row labels
    confusion_matrix_pd1= pd.DataFrame(conf_matrix_area, index=['actual wetlands (%s)'%units,'actual non-wetlands (%s)'%units], 
                                                                columns=['predicted wetlands (%s)'%units, 'predicted non-wetlands (%s)'%units])
    confusion_matrix_pd= confusion_matrix_pd1.round(3)
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
    
    #get name of composite for accuracy file name
#    if composite==None:
#        file_name='sklearn_accuracy_metrics'
#    else:
    file_name=ntpath.basename(composite).split('.')[0]
    
    print 'The accuracy metrics file is called: '
    print file_name
    
    print 'writing to files'
    writer=pd.ExcelWriter('C:\Users\Linnea\Documents\GitHubWetlands\wetland_identification\Post_Classification\Accuracy_TRAIN10_CW02-1{}.xlsx'.format(file_name))
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


def main(composite, true_ds):
    
    X_train, X_test, y_train, y_test, img= tt_split(composite_input, true_input)
    pred_class, f_imp= classify(X_train, X_test, y_train, img)
    pred_clean, true_clean=clean_data(y_test,pred_class)
    a,b,c,d,e= get_acc(pred_clean, true_clean)
    #in order to include units, define here:
    #there may be a way to do this from gdal metadata
    write_acc(a,b,c,d,e,f_imp,composite)

if __name__== '__main__':
    time0= time.time()
    main(composite_input, true_input)
    time1=time.time()
    print 'runtime was: {time} seconds'.format(time=(time1-time0))

