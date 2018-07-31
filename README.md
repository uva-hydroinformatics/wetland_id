# wetland_identification

The wetland identification model is an original open source, automated workflow consisting of three main parts: preprocessing, input variable calculation, and classification and accuracy assessment. Input data required to execute the model includes high-resolution DEM data and wetland delineations to serve as verification data, both in geotiff format. The main outputs from the model are geotiff wetland predictions and an accuracy report printed to an Excel file. In the preprocessing phase, the input DEM is first smoothed and then conditioned. Both the smoothed DEM (DEM (S)) and the smoothed, conditioned DEM (DEM (S, C)) are used for input variable calculation, where topographic wetness index (TWI), curvature, and cartographic depth-to-water index (DTW) grids are created. The input variables are merged into a multiband geotiff, where each band contains the information of a unique variable, and then passed to the classification and accuracy assessment. Before RF classification takes place, input wetland delineations are randomly split into training and testing subsets. Training data are used in conjunction with the merged input variables to train the RF model, producing a set of wetland predictions. Lastly, the testing data are used to assess the accuracy of predictions and an accuracy report is created summarizing model performance. This workflow is implemented in Python and executed using GDAL, SciPy, GRASS GIS, Scikit-Learn, and PyGeoNet.

wetland_id_defaults.py:

wetland_id_processing.py:

wetland_tool_grass.py:

filtering.py:

create_inputs.py:

class_acc_funcs.py:

raster_array_funcs.py:

Dependencies:
