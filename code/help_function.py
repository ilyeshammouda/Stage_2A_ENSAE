""" 
Created on Mon Jul 11:47:35 2023
@ author: ilyeshammouda
This code defines some useful functions that will be used 
"""

import numpy as np
from sklearn import linear_model
import random
from math import log,sqrt
from scipy.optimize import minimize

class Help_function(object):
    def __init__(self,Z,Y,threshold,cv=5):
        self.Z=Z
        self.Y=Y
        self.cv=cv
        self.threshold=threshold
    
    


