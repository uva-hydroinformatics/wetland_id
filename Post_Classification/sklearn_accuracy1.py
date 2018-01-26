# -*- coding: utf-8 -*-

"""
Created on Fri Jan 19 21:47:05 2018

@author: Linnea
"""


import sys
from sklearn import metrics, svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from osgeo import gdal, gdal_array, ogr
import numpy as np
import pandas as pd
from io import StringIO
import re
#from numpy import *
#from sklearn_test.py import test_arr
#from sklearn_test.py import clipped


pred= r"C:\Users\Linnea\Downloads\sklearn_files\sklearn_RF_geo6_clip.tif"
true_ds= r"C:\Users\Linnea\Downloads\sklearn_files\p06_vals_str_clip.tif"

#pred_array= gdal_array.###########
#true_array= gdal_array.###########

pred_gdal = gdal.Open(pred, gdal.GA_ReadOnly)

true_gdal = gdal.Open(true_ds, gdal.GA_ReadOnly)



#create predicted and true arrays from images
pred_array = pred_gdal.ReadAsArray().astype('int')

true_array = true_gdal.ReadAsArray().astype('int')

print type(pred_array)

print np.shape(pred_array)
print np.shape(true_array)

#make 1 dimensional so that they can be compared
pred_array1d= pred_array.flatten()

true_array1d= true_array.flatten()

#eliminate NaN pixels
pred_arr_vals1 = pred_array1d[pred_array1d != 255]
true_arr_vals1 = true_array1d[pred_array1d != 255]
true_arr_vals = true_arr_vals1[true_arr_vals1 != 3]
pred_arr_vals = pred_arr_vals1[true_arr_vals1 != 3]
#pred_arr_vals= pred_array_vals1[pred_array1d != 3]
#pred_arr_vals= pred_array_vals1[true_array1d != 3]

print pred_arr_vals
print true_arr_vals


acc_score= accuracy_score(true_arr_vals, pred_arr_vals)
results= confusion_matrix(true_arr_vals, pred_arr_vals)

#print 'TYPE IS: '
#print type(acc_score)
#print type(results)


#convert to sq km
results= results.astype(float)
print results
results*=0.000762


acc_score_pd= pd.Series(acc_score, index=['Fraction of correctly predicted points:'])
print acc_score_pd

#add column and row labels
## look at sklearn documentation to determine proper labels (what does the confusion matrix put out?)
results_pd= pd.DataFrame(results, index=['predicted wetlands (sq km)', 'predicted non-wetlands(square km)'], columns=['actual wetlands (square km)', 'actual non-wetlands (square km)'])


#save to a csv
results_pd.to_csv('C:\Users\Linnea\Downloads\sklearn_files\sklearn_accuracy_metrics\confusion_matrix1.csv')
acc_score_pd.to_csv('C:\Users\Linnea\Downloads\sklearn_files\sklearn_accuracy_metrics\score_accuracy.csv') #why can't file name start with a??

#000000000000000000000 http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
#returns same as attempt 3.

#classifier = svm.SVC(gamma=0.001)
#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(true_arr_vals, pred_arr_vals)))



#111111111111111111111111111111111111111111111111111111111111111111111111 ideas
 #classification report metric:
#classification=['Wetlands', 'Non-wetlands']
#creport= classification_report(true_array1d, pred_array1d)

##cr_split= creport.split('\n')
##float(u'creport')
#print 'creport:   '
#print creport
#print type(creport)
# 
#creport_pd=pd.DataFrame( creport, dtype=float)
# 
#creport_pd.to_csv('C:\Users\Linnea\Downloads\sklearn_files\sklearn_accuracy_metrics\class_report1.csv')


#22222222222222222222222222222222222 https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
 #support row is disfunctional. Check to see if these values should be expected. average column= .22???
#print 'entering classification report module'
#def class_report(report):
#    report_data = []
#    lines = report.split('\n') #report output is a string, needs splitting to parse
#    for line in lines[2:-3]: #"for each indicated word in the string:"
#        row = {}            #row is the dictionary
#        row_data = line.split('      ')
#        row['class'] = row_data[0]         #each word becomes a header
#        row['precision'] = float(row_data[1])
#        row['recall'] = float(row_data[2])
#        row['f1_score'] = float(row_data[3])
##        row['support'] = float(row_data[4])
#        report_data.append(row)
#    report_pd = pd.DataFrame.from_dict(report_data)
#    report_pd.to_csv('C:\Users\Linnea\Downloads\sklearn_files\sklearn_accuracy_metrics\class_report3.csv', index = False)
#    
##    
#report = classification_report(true_arr_vals, pred_arr_vals)
#print report
#class_report(report)


#333333333333333333333333333333 same link as above, returns nicely formatted csv but returns totally different
#numbers. average column wrong
def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

#txt report to df
report = metrics.classification_report(true_arr_vals, pred_arr_vals)
report_df = report_to_df(report)

#store, print, copy...
print (report_df)

report_df.to_csv('C:\Users\Linnea\Downloads\sklearn_files\sklearn_accuracy_metrics\class_report4.csv')


sys.exit(0)











#need the tiff files as a single dimensional numpy array in order to compare 
#for a binary confusion matrix
'''
pred_gdal = gdal.Open(pred, gdal.GA_ReadOnly)

true_gdal = gdal.Open(true, gdal.GA_ReadOnly)

pred_array= np.array(pred_gdal)

true_array= np.array(true_gdal)


ncol = pred_gdal.RasterXSize
nrow = pred_gdal.RasterYSize

ncolt = true_gdal.RasterXSize
nrowt = true_gdal.RasterYSize

print "Predicted ds Y size is: "
print nrow 

print "Predicted ds X Size is: "
print ncol

print "True ds Y size is: "
print nrowt 

print "True ds X Size is: "
print ncolt

#need to convert from dataset to array

#need to convert to km^2
'''

