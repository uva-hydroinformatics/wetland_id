# wetland_identification

The wetland identification model is an original open source, automated workflow consisting of three main parts: preprocessing, input variable calculation, and classification and accuracy assessment. Input data required to execute the model includes high-resolution DEM data and wetland delineations to serve as verification data, both in geotiff format. The main outputs from the model are geotiff wetland predictions and an accuracy report printed to an Excel file. In the preprocessing phase, the input DEM is first smoothed and then conditioned. Both the smoothed DEM (DEM (S)) and the smoothed, conditioned DEM (DEM (S, C)) are used for input variable calculation, where topographic wetness index (TWI), curvature, and cartographic depth-to-water index (DTW) grids are created. The input variables are merged into a multiband geotiff, where each band contains the information of a unique variable, and then passed to the classification and accuracy assessment. Before RF classification takes place, input wetland delineations are randomly split into training and testing subsets. Training data are used in conjunction with the merged input variables to train the RF model, producing a set of wetland predictions. Lastly, the testing data are used to assess the accuracy of predictions and an accuracy report is created summarizing model performance. This workflow is implemented in Python and executed using GDAL, SciPy, GRASS GIS, Scikit-Learn, and PyGeoNet (doi:10.1016/j.envsoft.2016.04.026) (see notes below on PyGeoNet usage). The wetland ID model is run by executing the main program of wetland_id_processing.py and input files, output paths, and parameters are changed by editing wetland_id_defaults.py.

Dependencies (other resources available for install instructions, steps we used listed below):
-Python 2.7
-TauDEM & GDAL
  TauDEM installation:
  1. Download TauDEM complete installer: http://hydrology.usu.edu/taudem/taudem5/
  2. Install GDAL with pip install .whl file from http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal (use the "cp27" to match python 2.7)
  3. Add environment variables
    3a. system variables: 
      "GDAL_DATA", value = "C:\Anaconda2\Lib\site-packages\osgeo\data\gdal"
      "MSIMPS_BIN", value = "C:\Program Files\Microsoft MPI\Bin"
      "MSMPI_INC", value = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\"
    3b. Add path environment variables (ordered top-down): 
      "C:\Program Files\GDAL"
      "C:\Anaconda2"
      "C:\Anaconda2\Library\bin"
      "C:\Anaconda2\Scripts"
      "C:\Program Files\Microsoft MPI\Bin\"
      "C:\Program Files (x86)\Microsoft SDKs\MPI"
      "C:\GDAL"
      "C:\Program Files\TauDEM\TauDEM5Exe"
      "C:\Anaconda2\Lib\site-packages"
      "C:\Anaconda2\Lib\site-packages\osgeo"
      "C:\Anaconda2\Lib\site-packages\osgeo\lib"
      "C:\Anaconda2\Lib\site-packages\osgeo\scripts"
-PyGeoNet & GRASS GIS
***Note: We use the version of PyGeoNet that was available in January, 2017 from https://sites.google.com/site/geonethome/ (also see related publications for this tool at this link). Some changes were made to this code to be executable for the wetland ID tool. Our changes did not alter the GeoNet workflow, and are reflected in the selected PyGeoNet modules posted with our code. Other PyGeoNet modules were unchanged but are necessary to run this workflow***
 PyGeoNet installation:
 1. Download available here https://sites.google.com/site/geonethome/source-code, Python version - latest update 2017/10/24
 2. Installation instructions and GRASS GIS download info available here https://sites.google.com/site/geonethome/howto, "Tutorial for installation of Python GeoNet v2 (pdf)"
   2a. Add system variables: 
       "GISBASE", value = "C:\Program Files\GRASS GIS 7.2.2"
       "GISRC", value = "C:\Users\USER\Documents\grassdata"
   2b. Add path environment variables (ordered top-down):
       "C:\Program Files\GRASS GIS 7.2.2"
       "C:\Program Files\GRASS GIS 7.2.2\etc\python\grass"
       "C:\Program Files\GRASS GIS 7.2.2\etc\python\grass\script"
       "C:\Program Files\GRASS GIS 7.2.2\bin"
       "C:\Program Files\GRASS GIS 7.2.2\scripts"
-Scikit-Learn & SciPy
  1. install scikit-learn >> pip install -U scikit-learn
  2. install scipy with pip install .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy (cp 27)
     
      
      
      
      
      
