import os
import numpy as np
import warnings
import pygeonet_prepare as Parameters
from pygeonet_rasterio import *
from pygeonet_plot import *
from pygeonet_nonlinear_filter import *
from pygeonet_slope_curvature import *
from pygeonet_flow_accumulation import *
from pygeonet_skeleton_definition import *
from pygeonet_fast_marching import *
from pygeonet_channel_head_definition import *
from pygeonet_network_delineation import *
from pygeonet_xsbank_extraction import *
import time
import wetland_id_defaults as default
"""EDITED: removed pygeonet nan removal method (L29-L35), main now takes one arg, a nanDemArray, which is created in 
the wetland id processing script using clean_data from raster_array_funcs.py

**if fast marching basin index "startpoint[x][y]" begins at [0][y], this is likely a sign that nan values are 
incorrectly filtered out...should be much faster process that starts at random [x][y] and does not go through 
entirety of len(x) len(y). If this is still happening and nan assignment is correct, GRASS may be trying to 
use corrupted files from previous runs --> we removed the check in pygeonet_grass that avoided re-doing grass calcs
"""


#---------------------------------------------------------------------------------
#------------------- MAIN FUNCTION--------------------------------------------------
#---------------------------------------------------------------------------------

def main(nanDemArray):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    print "Beginning GeoNet processing......................................... \n"
    pyg_t1 = time.time()
    print "Current working directory : ", os.getcwd()
    print "Reading input file path :",Parameters.demDataFilePath
    print "Reading input file :",Parameters.demFileName
#    DemArray = read_dem_from_geotiff(Parameters.demFileName, Parameters.demDataFilePath)

#    maskDemArray = raster_array_funcs.clean_array(DemArray)
##    nanDemArray = np.ma.masked_values(nanDemArray, defaults.demNanFlag)
#    nanDemArray = np.ma.filled(maskDemArray, np.nan)
#    nanDemArray[nanDemArray <= defaults.demNanFlag ] = np.nan #this is the original geonet coding but does not work when changed to ==
    defaults.figureNumber = 0
##    plt.switch_backend('agg')
    # plotting the raw DEM
    if defaults.doPlot == 1:
        raster_plot(nanDemArray, 'Input DEM')
    # Area of analysis
    Parameters.yDemSize=np.size(nanDemArray,0)
    Parameters.xDemSize=np.size(nanDemArray,1)
    # Calculate pixel length scale and assume square
    Parameters.maxLowerLeftCoord = np.max([Parameters.xDemSize, Parameters.yDemSize])
    print 'Raw DEM size: ',Parameters.yDemSize, 'rows x' ,Parameters.xDemSize, 'columns'
    # Compute slope magnitude for raw DEM and the threshold lambda used
    # in Perona-Malik nonlinear filtering. The value of lambda (=edgeThresholdValue)
    # is given by the 90th quantile of the absolute value of the gradient.
    edgeThresholdValue = lambda_nonlinear_filter(nanDemArray)
    
#    #*******************performing PM filtering using the anisodiff*********************
    print "Performing Perona-Malik nonlinear filtering......................... \n"

    t0 = clock()
    filteredDemArray = anisodiff(nanDemArray, defaults.nFilterIterations,
                                 edgeThresholdValue,
                                 defaults.diffusionTimeIncrement,
                                 (Parameters.demPixelScale,
                                  Parameters.demPixelScale), 2)
    t1 = clock()
    print "time taken to complete nonlinear filtering:",t1-t0," seconds"
    # plotting the filtered DEM
    if defaults.doPlot == 1:
        raster_plot(filteredDemArray, 'Filtered DEM')
    # Writing the filtered DEM as a tif
#    write_geotif_filteredDEM(filteredDemArray, Parameters.demDataFilePath, Parameters.demFileName)
    write_geotif_generic(filteredDemArray, Parameters.demDataFilePath, Parameters.pmGrassGISfileName.split('\\')[-1])
    
    #*********************Computing slope and curvature of filtered DEM******************
    #computing slope
    print 'Computing slope of filtered DEM'
    slopeDemArray = compute_dem_slope(filteredDemArray, Parameters.demPixelScale)
    # Writing the slope array
    outfilepath = Parameters.geonetResultsDir
    demName = Parameters.demFileName.split('.')[0]
    outfilename = demName +'_slope.tif'
    write_geotif_generic(slopeDemArray,outfilepath,outfilename)
    
    # Computing curvature
    print 'computing curvature of filtered DEM'
    curvatureDemArray, curvatureDemMean, \
                       curvatureDemStdDevn = compute_dem_curvature(filteredDemArray,
                                                                   Parameters.demPixelScale,
                                                                   defaults.curvatureCalcMethod)
    # Writing the curvature array
    outfilename = demName +'_curvature.tif'
    write_geotif_generic(curvatureDemArray,outfilepath,outfilename)
    # plotting the curvature image
    if defaults.doPlot == 1:
        raster_plot(curvatureDemArray, 'Curvature DEM')
    #Compute curvature quantile-quantile curve
    # This seems to take a long time ... is commented for now
    print 'Computing curvature quantile-quantile curve'
    #osm,osr = compute_quantile_quantile_curve(finiteCurvatureDemList)
    #print osm[0]
    #print osr[0]
    thresholdCurvatureQQxx = 1
    # have to add method to automatically compute the thresold
    # .....
    # .....

    # ***********************Computing contributing areas*************************
    print 'Computing upstream accumulation areas using MFD from GRASS GIS.........\n'
    """
    return {'outlets':outlets, 'fac':nanDemArrayfac ,\
            'fdr':nanDemArrayfdr ,'basins':nanDemArraybasins,\
            'outletsxxProj':outletsxxProj, 'outletsyyProj':outletsyyProj,\
            'bigbasins':allbasins}
    """
    #check if flow accumulation results exists within pygeonet_flow_accumulation.py
    # Call the flow accumulation function
    flowroutingresults = flowaccumulation(filteredDemArray)
    # Read out the flowroutingresults into appropriate variables
    outletPointsList = flowroutingresults['outlets']
    flowDirectionsArray = flowroutingresults['fdr']
    flowArray = flowroutingresults['fac']
    flowArray[np.isnan(filteredDemArray)]=np.nan
    flowMean = np.mean(flowArray[~np.isnan(flowArray[:])])
    print 'Mean upstream flow: ', flowMean
##    # These are actually not sub basins, if the basin threshold
##    # is large, then you might have as nulls, so best
##    # practice is to keep the basin threshold close to 1000
##    # default value is 10,000
##    #subBasinIndexArray = flowroutingresults['basins']
##
##    
##    #subBasinIndexArray[subBasinIndexArray==-9999]=np.nan
    basinIndexArray = flowroutingresults['bigbasins']
##
    # Define a skeleton based on flow alone
    skeletonFromFlowArray = \
    compute_skeleton_by_single_threshold(flowArray,
                                         defaults.flowThresholdForSkeleton)
    # Define a skeleton based on curvature alone
    skeletonFromCurvatureArray =\
    compute_skeleton_by_single_threshold(curvatureDemArray,
                                         curvatureDemMean+
                                         thresholdCurvatureQQxx*curvatureDemStdDevn)
    # Writing the skeletonFromCurvatureArray array
    outfilename = demName+'_curvatureskeleton.tif'
    write_geotif_generic(skeletonFromCurvatureArray,\
                         outfilepath, outfilename)
    # Define a skeleton based on curvature and flow
    skeletonFromFlowAndCurvatureArray =\
    compute_skeleton_by_dual_threshold(curvatureDemArray, flowArray, \
                                       curvatureDemMean+thresholdCurvatureQQxx*curvatureDemStdDevn, \
                                       defaults.flowThresholdForSkeleton)
    # plot the skeleton with outlets
    if defaults.doPlot == 1:
        raster_point_plot(skeletonFromFlowAndCurvatureArray,outletPointsList,
                          'Skeleton with outlets',cm.binary)
    # Writing the skeletonFromFlowAndCurvatureArray array
    outfilename = demName+'_skeleton.tif'
    write_geotif_generic(skeletonFromFlowAndCurvatureArray,\
                         outfilepath,outfilename)
#    sys.exit(0)
    # Making outlets for FMM
    print type(outletPointsList)
    print outletPointsList
    fastMarchingStartPointListFMM = Fast_Marching_Start_Point_Identification(outletPointsList, basinIndexArray)
    # Computing the local cost function
    print 'Preparing to calculate cost function'
    curvatureDemArray = Curvature_Preparation(curvatureDemArray)
    # Calculate the local reciprocal cost (weight, or propagation speed in the
    # eikonal equation sense).  If the cost function isn't defined, default to
    # old cost function.
    print 'Calculating local costs'
    reciprocalLocalCostArray = Local_Cost_Computation(flowArray, flowMean,
                                                      skeletonFromFlowAndCurvatureArray,
                                                      curvatureDemArray)
    geodesicDistanceArray = Fast_Marching(fastMarchingStartPointListFMM,
                                          basinIndexArray, flowArray,
                                          reciprocalLocalCostArray)
    if defaults.channelheadPredefined == 0:
        # Locating end points
        print 'Defining channel heads'
        xx, yy = Channel_Head_Definition(skeletonFromFlowAndCurvatureArray,
                                         geodesicDistanceArray)
    else:
        print 'Using given channel heads'
        channelhead_filename = demName+'_channelHeads.tif'
        channelheadArray = read_geotif_generic(outfilepath, channelhead_filename)
        channelheadArray = np.where(channelheadArray==1)
        xx = channelheadArray[1]
        yy = channelheadArray[0]
    geodesicPathsCellDic, numberOfEndPoints = Channel_Definition(xx,yy, geodesicDistanceArray, basinIndexArray, flowDirectionsArray)
    create_xs_bank_wse(filteredDemArray,slopeDemArray,numberOfEndPoints,geodesicPathsCellDic)
    print 'Finished pyGeoNet'
    pyg_t2 = time.time()
    print "Pygeonet channels created, execution time: %.2f" %(pyg_t2 - pyg_t1)
    return Parameters.drainagelineFileName
    
if __name__ == '__main__':
    t0 = clock()
    main(nanDemArray)
    t1 = clock()
    print "time taken to complete the script is::",t1-t0," seconds"
    print "script complete"
