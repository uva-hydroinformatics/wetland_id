# wetland_identification

The wetland identification model is an original open source, automated workflow consisting of three main parts: preprocessing, input variable calculation, and classification and accuracy assessment. Input data required to execute the model includes high-resolution DEM data and wetland delineations to serve as verification data, both in geotiff format. The main outputs from the model are geotiff wetland predictions and an accuracy report printed to an Excel file. In the preprocessing phase, the input DEM is first smoothed and then conditioned. Both the smoothed DEM (DEM (S)) and the smoothed, conditioned DEM (DEM (S, C)) are used for input variable calculation, where topographic wetness index (TWI), curvature, and cartographic depth-to-water index (DTW) grids are created. The input variables are merged into a multiband geotiff, where each band contains the information of a unique variable, and then passed to the classification and accuracy assessment. Before RF classification takes place, input wetland delineations are randomly split into training and testing subsets. Training data are used in conjunction with the merged input variables to train the RF model, producing a set of wetland predictions. Lastly, the testing data are used to assess the accuracy of predictions and an accuracy report is created summarizing model performance. This workflow is implemented in Python and executed using TauDEM, GDAL, SciPy, GRASS GIS, Scikit-Learn, and PyGeoNet (doi:10.1016/j.envsoft.2016.04.026) (see notes below on PyGeoNet usage). The wetland ID model is run by executing the main program of wetland_id_processing.py and input files, output paths, and parameters are changed by editing wetland_id_defaults.py.


## Dependencies

* Python 2.7
* [TauDEM](http://hydrology.usu.edu/taudem/taudem5/index.html) 
* [GDAL](https://www.gdal.org/)
* [SciPy](https://www.scipy.org/)
* [GRASS GIS](https://grass.osgeo.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [PyGeoNet](https://sites.google.com/site/geonethome/)

### Installing
**TauDEM and GDAL**

1. Download TauDEM complete installer: http://hydrology.usu.edu/taudem/taudem5/

2. Install GDAL for Python with [.whl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
```
>> pip install *.whl 
```


3. Add environment variables

system variables:
```
"GDAL_DATA", value = "C:\Anaconda2\Lib\site-packages\osgeo\data\gdal"
"MSIMPS_BIN", value = "C:\Program Files\Microsoft MPI\Bin"
"MSMPI_INC", value = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\"
```
 
 path environment variables (ordered top-down): 
 ```
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
```

**Scikit-Learn**
```
  >> pip install -U scikit-learn
  ```
  
  **SciPy with [.whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)**
 ```
  >> pip install *.whl
  ```

**PyGeoNet**
_Note: We use PyGeoNet Version 2.0, obtained January, 2017 from https://sites.google.com/site/geonethome/ \[1; 2; 3; 4\]. Some changes were made to this code to be executable for the wetland ID tool. PyGeoNet scripts that have been edited for use within the wetland ID tool are annotated within first few lines, our changes did not alter the GeoNet workflow. PyGeoNet scripts without 'edit' notes reflect the original versions posted with Version 2.0, described by \[1\]._
 
 1. Download available here https://sites.google.com/site/geonethome/source-code, Python version - latest update 2017/10/24
 2. Installation instructions and GRASS GIS download info available here https://sites.google.com/site/geonethome/howto, "Tutorial for installation of Python GeoNet v2 (pdf)"
 3. Add system variables:
 ```
       "GISBASE", value = "C:\Program Files\GRASS GIS 7.2.2"
       "GISRC", value = "C:\Users\USER\Documents\grassdata"
  ```
   Add path environment variables (ordered top-down):
   ```
       "C:\Program Files\GRASS GIS 7.2.2"
       "C:\Program Files\GRASS GIS 7.2.2\etc\python\grass"
       "C:\Program Files\GRASS GIS 7.2.2\etc\python\grass\script"
       "C:\Program Files\GRASS GIS 7.2.2\bin"
       "C:\Program Files\GRASS GIS 7.2.2\scripts"
   ``` 
   4. Install GRASS r.stream.basins:
   ```
   open cmd as admin
   >> grass72 
   >> g.extension r.hydrodem -s (where -s flag  installs for all users)
```

## Authors

* **Gina O'Neil**
* **Linnea Saby**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* PyGeoNet sources:  

1. Sangireddy, H.*, C.P. Stark, A. Kladzyk*, Passalacqua, P. (2016). GeoNet: An open source software for the automatic and objective extraction of channel heads, channel network, and channel morphology from high resolution topography data, Environmental Modeling and Software, 83, 58-73, doi:10.1016/j.envsoft.2016.04.026.  

2. Passalacqua, P., P. Belmont, and E. Foufoula-Georgiou (2012), Automatic channel network and geomorphic feature extraction in flat and engineered landscapes, Water Resour. Res., 48, W03528, doi:10.1029/2011WR010958.  

3. Passalacqua, P., P. Tarolli, and E. Foufoula-Georgiou (2010), Testing space-scale methodologies for automatic geomorphic feature extraction from lidar in a complex mountainous landscape, Water Resour. Res., 46, W11535, doi:10.1029/2009WR008812.  

4. Passalacqua, P., T. Do Trung, E. Foufoula-Georgiou, G. Sapiro, and W. E. Dietrich (2010), A geometric framework for channel network extraction from lidar: Nonlinear diffusion and geodesic paths, J. Geophys. Res., 115, F01002, doi:10.1029/2009JF001254.

  
