'''
This module contains various test functions for the ZORO algorithm. 
All of them exhibit gradient sparsity or compressibility.

'''
import numpy as np
#import sys
import math

class SparseQuadric(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.s = s
        self.dim = n
        self.rng = np.random.RandomState()
        
    def __call__(self,x):
        f_no_noise = np.dot(x[0:self.s],x[0:self.s])
        return f_no_noise + self.noiseamp*self.rng.randn()

    
class MaxK(object):
    '''An implementation of the max-k-squared-sum function.'''
    def __init__(self, n, s, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.dim = n
        self.s = s
        self.rng = np.random.RandomState()

    def __call__(self, x):

        idx = np.argsort(np.abs(x))
        idx2 = idx[self.dim-self.s:self.dim]
        f_no_noise = np.dot(x[idx2], x[idx2])/2
        return f_no_noise + self.noiseamp*self.rng.randn()
 

class CompressibleQuadric(object):
    '''An implementation of the sparse quadric function.'''
    def __init__(self, n, decay_factor, noiseamp):
        self.noiseamp = noiseamp/np.sqrt(n)
        self.decay_factor = decay_factor
        self.dim = n
        self.rng = np.random.RandomState()
        self.diag = np.zeros(n)
        for i in range(0,n):
            self.diag[i] = math.exp(-self.decay_factor*i)
        
    def __call__(self,x):
        #f_no_noise = 0
        #for i in range(0,self.dim):
            #f_no_noise += math.exp(-self.decay_factor*i)*x[i]**2
        f_no_noise = np.dot(self.diag * x, x)
        return f_no_noise + self.noiseamp*self.rng.randn()
        #f_no_noise = np.dot(x[0:self.s],x[0:self.s])
        #f_no_noise += 1e-4*np.dot(x[self.s:self.dim],x[self.s:self.dim])
        #return f_no_noise + self.noiseamp*self.rng.randn()
    

class norm_with_a_Gaussian_matrix(object):
    '''An implementation of a weakly sparse  vector.'''
    def __init__(self,n, x_star, noiseamp,A):
        self.noiseamp = noiseamp
        self.dim = n
        self.x_star=x_star
        self.rng = np.random.RandomState()
        self.A=A

    def __call__(self, x):
        A=self.A
        f_no_noise = (np.linalg.norm(A@(x-self.x_star))/self.dim)**2
        return f_no_noise + self.noiseamp*self.rng.randn()
    


class square_of_the_difference_support_S(object):
        def __init__(self,n, x_star, noiseamp):
            self.noiseamp = noiseamp
            self.dim = n
            self.x_star=x_star
            self.rng = np.random.RandomState()
        
        def __call__(self, x):
            s=np.count_nonzero(self.x_star)
            x_minus_x_star_square=np.square(x-self.x_star)
            f_no_noise=np.sum(x_minus_x_star_square[:s])
            return f_no_noise + self.noiseamp*self.rng.randn()
        

