# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:30:46 2024

@author: Oaklin Keefe, new function based on Houria Madani's MATLAB equivalent code

This model_bkgnd function fits the background of the given frame, 
ignoring bright spots (cities), as a polynomial with a step in the 
middle of the frame. We have substituted the numpy.polyfit and 
numpy.polyval fitting functions for a long-form linear algebra 
least-squares fit to account for the step that interrupts the 
polynomial fit.

"""

import numpy as np

def model_bkgnd(frame, dqf, degree, step, nsig):
    # Eliminate NaNs
    tmp = frame 
    tmp[np.isnan(tmp)] = 0 #replace NaNs with 0 
    
    # Integrate over spectral bins
    test = np.sum(tmp, axis=1) 
    
    
    # Find bright locations (cities) in top and bottom of frame w/ nsig MAD
    city_top = np.zeros((1024,1))
    z = np.abs(test[:1024] - np.median(test[:1024]))
    mad = np.median(z)
    city_top[z > nsig * 1.4826 * mad] = 1    # 1.4826 is the ratio of sigma to median absolute difference for a normal distribution
    
    city_bot = np.zeros((1024,1))
    z = np.abs(test[1024:] - np.median(test[1024:]))
    mad = np.median(z)
    city_bot[z > nsig * 1.4826 * mad] = 1 
    
    city = np.vstack((city_top, city_bot)).reshape((2048,)) 
    city_not = np.logical_not(city)
    

    # Independent variables for fit
    x = ((np.arange(0, tmp.shape[0]))-tmp.shape[0]/2+0.5)/tmp.shape[0] 

    # Allocate memory for background frame
    bkgnd = np.full(tmp.shape, np.nan) 

    # Model coficients
    p = np.zeros((degree+1+step, tmp.shape[1])) 

    # Design a matrix for fitting with polynomial and optionally a step
    F = np.zeros((len(x), degree+1+step)) 
    for n in range(0, degree+1):
        F[:, n] = x**n 
    if step:
        F[:1024, degree+1] = -0.5 
        F[1024:, degree+1] = 0.5  
    
    # Backgrouund fit for each spectral bin
    for j in range(tmp.shape[1]):
        # Use only nominal data quality and not bright locations
        dqf_nominal = dqf[:,j]==0
        sel = np.logical_and(city_not, dqf_nominal)
        
        # Skip background fitting if not enough useful data
        if float(np.count_nonzero(sel))<(0.5*float(sel.size)):
            continue
        
        F_prime = np.conjugate((F[sel,:]).T)
        A = (F_prime) @ (F[sel,:]) 
        v = (F_prime) @ (tmp[sel,j])
        p[:,j] = np.linalg.solve(A, v)  
        
        # Evaluate fit for the background
        bkgnd[:,j] = F @ p[:,j]
    
    # Subtract background from frame
    signal = frame - bkgnd
    
    return signal, bkgnd, p