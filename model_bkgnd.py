# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:40:32 2024

@author: jwelsh
History of modifications
06/06/2024: Houria Madani
Replaced bad = np.isnan(data) by  np.isnan(y) to avoid the 
crashing of np.polyfit.  
The array x(~bad) appears empty when using np.isnan(data) for some data sets.

"""

import numpy as np

def model_bkgnd(data, degree, nsig):
    x = np.arange(1, len(data) + 1)
    y = np.array(data)

    # bad is part of the signal that is not background
    bad = np.isnan(y)
    nbad = np.sum(bad)
    
    for n in range(5):
        if nbad > (len(x) - degree - 1):
            signal = np.full(np.shape(x), np.nan)
            bkgnd = signal
            return signal

        # Fit polynomial only to non-NaN values
        p = np.polyfit(x[~bad], y[~bad], degree)
        # Evaluate polynomial at all points (including NaNs)
        bkgnd = np.polyval(p, x)
        # Calculate signal
        signal = data - bkgnd
        # Identify cities by their fit residuals relative to the background
        z = np.abs(signal - np.median(signal))
        mad = np.median(z[~bad])
        # 1.4826 is the ratio of sigma to median absolute difference for
        # a normal distribution
        bad[z > nsig * 1.4826 * mad] = 1
        nbad0 = np.sum(bad)
        if nbad0 <= nbad:
            break
        nbad = nbad0
    return signal