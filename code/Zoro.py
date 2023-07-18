""" 

This code was inspired from the article: https://arxiv.org/abs/2003.13001
HanQin Cai, Daniel Mckenzie, Wotao Yin, and Zhenliang Zhang. Zeroth-Order Regularized Optimization (ZORO): 
Approximately Sparse Gradients and Adaptive Sampling. arXiv preprint arXiv: 2003.13001.
As well as their git repo: https://github.com/caesarcai/ZORO
Some changes have been made to test new algorithms that can be more adaptive than the ones presented in the article.
"""
import numpy as np
import numpy.linalg as la
from interface import BaseOptimizer
from Cosamp import cosamp
from help_function import Lasso_reg
from help_function import debiased_Lasso
from help_function import IHT_classique
from help_function import IHT_ad


class ZORO(BaseOptimizer):
    '''
    ZORO for black box optimization. 
    '''
    def __init__(self, x0, f, params, algo ,threshold_IHT=2,function_budget=10000, prox=None,
                 function_target=None,s=20,step_IHT=0.0000001,itt_IHT=30,C_IHT=0.9,lamda_IHT=0.1):
        
        super().__init__()
        
        self.function_evals = 0
        self.function_budget = function_budget
        self.function_target = function_target
        self.f = f
        self.x = x0
        self.n = len(x0)
        self.algo=algo
        self.t = 0
        self.delta = params["delta"]
        self.sparsity = params["sparsity"]
        self.step_size = params["step_size"]
        self.num_samples = params["num_samples"]
        self.prox = prox
        self.s=s
        self.step_IHT=step_IHT
        self.itt_IHT=itt_IHT
        self.threshold_IHT=threshold_IHT
        self.C_IHT=C_IHT
        self.lamda_IHT=lamda_IHT

        # Define sampling matrix
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1

        cosamp_params = {"Z": Z, "delta": self.delta, "maxiterations": 10,
                         "tol": 0.5, "sparsity": self.sparsity}
        self.cosamp_params = cosamp_params



    # Handle the (potential) proximal operator
    def Prox(self, x):
        if self.prox is None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
        


       
    def GradEstimate(self):
        '''
        Gradient estimation sub-routine.
        '''
      
        maxiterations = self.cosamp_params["maxiterations"]
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        sparsity = self.cosamp_params["sparsity"]
        tol = self.cosamp_params["tol"]
        num_samples = np.size(Z, 0)
        x = self.x
        f = self.f
        y = np.zeros(num_samples)
        function_estimate = 0
        
        for i in range(num_samples):
            y_temp = f(x + delta*np.transpose(Z[i,:]))
            y_temp2 = f(x)
            function_estimate += y_temp2
            y[i] = (y_temp - y_temp2)/(np.sqrt(num_samples)*delta)
            self.function_evals += 2
            
        function_estimate = function_estimate/num_samples
        
        Z = Z/np.sqrt(num_samples)
        if self.algo=='CoSaMP':
            grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
        if self.algo=='Lasso':
            grad_estimate=Lasso_reg(y,Z)
        if self.algo=='DLasso':
            grad_estimate=debiased_Lasso(y,Z,delta)
        if self.algo=='IHT_Classique':
            grad_estimate=IHT_classique(X=Z,Y=y,s=self.s,step=self.step_IHT,max_iterations=self.itt_IHT)
        if self.algo=='IHT_ad':
            grad_estimate=IHT_ad(X=Z,Y=y,threshold=self.threshold_IHT,C=self.C_IHT,step=self.step_IHT,max_iterations=self.itt_IHT,lamda=self.lamda_IHT)
        return grad_estimate, function_estimate
    

    

    def step(self):
        '''
        Take step of optimizer
        '''
   
        grad_est, f_est = self.GradEstimate()
        self.fd = f_est
        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        self.x = self.Prox(self.x -self.step_size*grad_est) # gradient descent 

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals, self.x, 'B'

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.x, 'T'
 
        self.t += 1
        return self.function_evals, False, False
    


    def Zoro(self):
        performance_log_ZORO = [[0, self.f(self.x)]]
        termination = False
        while termination is False:
            evals_ZORO, solution_ZORO, termination = self.step()
            # save some useful values
            performance_log_ZORO.append( [evals_ZORO,np.mean(self.fd)] )
            # print some useful values
            self.report( 'Estimated f(x_k): %f  function evals: %d\n' %
            (np.mean(self.fd), evals_ZORO) )
        return(performance_log_ZORO)

