#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  8 10:47:57 2024

@author: Jason Welsh

History of Modifications
09/26/2024   Oaklin Keefe             
             Streamlined code for determining wavelength based on VIS or UV, or
             another user-specific use-case.
             Updated background subtraction section to run model_bkgnd function
             on the entire frame.

             
06/12/2024   Houria Madani
             Added saving of the city variable to a netCDF file
             or a csv file if desired.
             Made changes to speed up the code; this includes removel of
             unecessary big arrays.
             Changed the numerical integration to use trapz
             This sped up the script and it now takes less than minutes to run 
             one granule of 101 frames instead of 50 minutes.
"""
#Import the necessary libraries into python
import matplotlib.pyplot as plt
from despeckle import despeckle
from destreak import destreak
from model_bkgnd import model_bkgnd 
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import os
import glob
import time
from scipy.integrate import trapz
import numpy.ma as ma
import pandas as pd

#Time the script
start_time = time.time()

# Configuration flags to save to files (True) or not (False)
save_to_netcdf = False
save_to_csv = False

# 'wvla' and 'wvlb' are used in the variable names to access the appropriate variables for VIS or UV
# The user can specify their own values depending on their use-case (e.g. VIS, UV, moonlight, etc.)
# this case here is for VIS
# for UV use wvla = 290 and wvlb = 490
VIS_case = True
if VIS_case == True:
    UV_case= False
    wvla = 540
    wvlb = 740
else:
    UV_case = True
    wvla = 290
    wvlb = 490

# frame clean up configuration
nsig = 10   # number of MAD filter sigmas to find hot pixels
nsig1 = 6   # number of MAD filter sigmas to identify bad quadrants
nsig2 = 6   # number of MAD filter sigmas to despeckle the frame
nsig3 = 3   # number of MAD filter sigmas for background subtraction 
degree = 2  # degree of polynomial for background modeling
persistence_for_a_hot_pixel = 3  # number of required repetitions to be persistent

# Define the directory path of where to retrieve all netcdf files
# Note that this version of the code works with a directory where there is 
# a single granule

directory = r'C:\AIST NN Project 2024\city_lights_code_for_TEMPO\one_granule3' 

# Find all NetCDF files in the directory and its subdirectories
nc_files = glob.glob(os.path.join(directory, '**', '*.nc'), recursive=True)
filename = nc_files[0]

# Process the one granule from the directory
with Dataset(filename, 'r') as f:
    # Read the data from the radiance group
    rad = f.groups['band_'+str(wvla)+'_'+str(wvlb)+'_nm']['radiance'][:]
    
print("Finished reading radiance")

# Convert masked 3D array to numpy array without mask
numpy_array_3d = ma.getdata(rad)
rad = numpy_array_3d
del(numpy_array_3d)
num_mirror_steps = rad.shape[0]
num_spatial_samples = rad.shape[1]
num_lambdas = rad.shape[2]

# Find the Hot pixels and bad quadrants and find the anomalously oversubtracted quadrants
mquad = np.zeros((2, num_mirror_steps))
        
# Start processing a frame; each mirror step is a frame
print("Start processing a frame; each mirror step is a frame")
for n in range(num_mirror_steps):
    frame = rad[n, :, :]
    # Compute mean of top of focal plane
    tmp = frame[0:1024, :]
    mquad[0, n] = np.mean(tmp)
    # Compute mean of bottom of focal plane
    tmp = frame[1024:2048, :]
    mquad[1, n] = np.mean(tmp)
    
print("Identify bad quadrants")        
# Identify bad quadrants where a quadrant is a spectral band in 
# the top or bottom of the focal plane
# 1.4826 is the ratio of sigma to median absolute difference for
# a normal distribution
x = np.abs(mquad - np.median(mquad, axis=1)[:, np.newaxis])
mad = np.median(x, axis=1)[:, np.newaxis]
bad = np.abs(x) > nsig1 * 1.4826 * mad

print("Find persistently hot pixels")
# Find persistently hot pixels using despeckle
hot_pix = np.zeros((num_spatial_samples, num_lambdas))
for n in range(num_mirror_steps):
    frame = rad[n, :, :]
    _, sf2 = despeckle(frame,None, nsig)
    hot_pix = hot_pix + sf2
    
print("Clean up the radiance")
# Clean up the radiance by despeckle, destreak, and background subtraction
for j in range(num_mirror_steps):
    frame = rad[j, :, :]
    # Despeckle/Destreak frame
    _, qf0 = despeckle(frame, hot_pix > persistence_for_a_hot_pixel, nsig2)
    frame = destreak(frame, qf0)
    
    # Make quality flag (qf) for bad quadrants
    if bad[0, j]:
        qf0[0:1024, :] = 3
    if bad[1, j]:
        qf0[1024:2048, :] = 3

    # Apply qf to put NaN in place of bad pixels or mask them out
    frame[qf0 != 0] = np.nan

    # Mask edge rows
    frame[0:5, :] = np.nan
    frame[2042:, :] = np.nan
    
    # Background Subtraction 
    bkgnd_deg = 3  
    bkgnd_step = 1 
    signal, bkgnd, p = model_bkgnd(frame, qf0, bkgnd_deg, bkgnd_step, nsig3)
        
    # Save the cleaned up frame in the rad variable
    rad[j, :, :] = signal


end_time1 = time.time()
elapsed_time1 = (end_time1 - start_time)/60
print("Elapsed time1:", elapsed_time1, "minutes")


# Wavelength range for numerical integration
# This is for the VIS case; for the UV case use 290 to 490
# Here, we make the UV case use 390-490 to match the correspoinding MATLAB code
if VIS_case == True:
    UV_case= False
    wvl1 = 540
    wvl2 = 640
else:
    UV_case = True
    wvl1 = 390
    wvl2 = 490

# Use the cleaned up radiance from the previous processing and 
# read the nominal wavelengths from the input netCDF file
print("read the nominal wavelengths ")
with Dataset(filename, 'r') as f:
    # Read the wavelengths from the selected radiance group
    wvl = f.groups['band_'+str(wvla)+'_'+str(wvlb)+'_nm']['nominal_wavelength'][:]

print("Generate a city lights map ")
# Generate a city lights map 
city = np.zeros((num_spatial_samples, num_mirror_steps))  # Initialize city array

# average wavelengths 
lam2=np.mean(wvl,0)

# delete arrays no longer needed
del(wvl)
del(mquad)
del(hot_pix)
del(frame)

# J_per_photon is Joules per photon computed using Planck's constant
# and speed of light
J_per_photon = 6.62607015e-34 * 299792458 / (lam2 * 1e-9)

for n in range(num_lambdas):  
    rad[:,:,n] = rad[:,:,n]*J_per_photon[n];

# delete arrays no longer needed
del(J_per_photon)   
del(qf0)
del(sf2) 
del(tmp)
    
end_time2 = time.time()

elapsed_time2 = (end_time2 - start_time)/60

print("Elapsed time2:", elapsed_time2, "minutes")

# select the wavelength range for integration
good_indices = np.where((lam2 >= wvl1) & (lam2 <= wvl2))[0]
if len(good_indices) > 0:
    # Select the first and last non-NaN members
    wvl_start = good_indices[0]
    wvl_stop = good_indices[-1] + 1
lam4=lam2[wvl_start:wvl_stop]

 # Loop over the number of mirror steps (i.e. frames)
for n in range(num_mirror_steps):     
    # Trapezoidal numerical integration over wavelengths
    city[:,n] = trapz(rad[n, :, wvl_start:wvl_stop],lam4)
# Convert to nanowatts
city=city*1e9

#Plot out a colormap of the city processed data variable        
plt.pcolor(city,vmin=0, vmax=100,cmap='viridis');plt.colorbar()


# Save to csv file if desired
if save_to_csv:
    # Extract the filename using os.path.basename
    file_name = os.path.basename(filename)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(file_name)
    
    # Append '_output' to the name
    file_name_output = name + '_output' + '.csv'

    # reshape the city array
    array_2d = city.reshape(city.shape[0], -1)

    # Convert the 2D array to a DataFrame
    df = pd.DataFrame(array_2d)

    # Save the DataFrame to a CSV file
    df.to_csv(file_name_output)

end_time3 = time.time()
elapsed_time3 = (end_time3 - start_time)/60
print("Elapsed time3:", elapsed_time3, "minutes")


# Save the city data in a new NetCDF file if desired
if save_to_netcdf:
    # Extract the filename using os.path.basename
    file_name = os.path.basename(filename)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(file_name)
    
    # Append '_output' to the name
    file_name_output = name + '_output' + ext
     
    ncfile = Dataset(file_name_output, 'w', format='NETCDF4')
    
    # Create dimensions
    ncfile.createDimension('dim1', city.shape[0])
    ncfile.createDimension('dim2', city.shape[1])
    
    # Create a variable to store the data
    nc_var = ncfile.createVariable('city', np.float32, ('dim1', 'dim2'))
    
    # Write data to the variable
    nc_var[:] = city
    
    # Add metadata
    ncfile.title = 'NetCDF File to save the city data for file ' + file_name
    
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ncfile.history = f'Created {current_time}'
    ncfile.date_created = current_time
    
    # Close the NetCDF file
    ncfile.close()    

# Print out the amount of time it took to process the entire script
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print("Elapsed time:", elapsed_time, "minutes")
