# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:47:57 2024

@author: Houria Madani
    This function is used to find persistently hot pixels
    or despeckle the frame
"""
import numpy as np
def despeckle(arg1, arg2=None, arg3=None):
    frame=arg1
    nsig=arg3
    
    sf = np.zeros_like(frame)

    for n in range(frame.shape[0]):
        x = np.abs(frame[n, :] - np.median(frame[n, :]))
        if arg2 is not None:
            # despeckle the frame without considering the hot pixels
            hot = arg2
            mad = np.median(x[~hot[n, :]])
            # 1.4826 is the ratio of sigma to median absolute difference for
            # a normal distribution
            frame[n, x > nsig * 1.4826 * mad] = 0
            sf[n, x > nsig * 1.4826 * mad] = 2

        else:
            # Find persistently hot pixels
            mad = np.median(x)

            frame[n, np.abs(x) > nsig * 1.4826 * mad] = 0

            sf[n, np.abs(x) > nsig * 1.4826 * mad] = 1

    if arg2 is not None:
        sf[hot] = 1
        
    return frame, sf
