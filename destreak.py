# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:25:45 2024

@author: jwelsh
"""
import numpy as np

def destreak(frame, qf):
    for n in range(frame.shape[1]):
        tmp = frame[:1024, n]
        frame[:1024, n] = tmp - np.median(tmp[qf[:1024, n] == 0])
        
        tmp = frame[1024:2048, n]
        frame[1024:2048, n] = tmp - np.median(tmp[qf[1024:2048, n] == 0])
        
    return frame