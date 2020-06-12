# palm_python
Read and process model output of the PALM model

## scripts
Located in the palm_python/ root directory, there are example scripts which call functions located in the module and do some useful tasks as:

* deliver raw timeseries of (non-)dimensionalized values
* deliver vertical profiles of horizontally averaged flow measures
* deliver vertical and horizontal crosssections of all three velocity components
* calculate temporal autocorrelations 
* calculate energy density spectra
* calculate turbulence intensities of horizontal velocity components
* calculate integral length scales of the flow

## modules

* /palm_py/env/ 
  * creates environment for palm_python data output
* /palm_py/read/  
  * read PALM-model output data (netcdf) and wind tunnel output data
* /palm_py/calc/
  * calculates flow measures
* /palm_py/plot/
  * plots flow measures
