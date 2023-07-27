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
from help_function import ISTA_ad,IHT_ad,IHT_classique,debiased_Lasso,Lasso_reg
import projection as proj
from MD import AcceleratedMethod
import warnings

class ZORO_MD(BaseOptimizer):
    '''
    ZORO for black box optimization. 
    '''
    def __init__(self, x0, f, params, algo ,threshold_IHT=2,function_budget=10000, prox=None,
                 function_target=None,s=20,step_IHT=0.0000001,itt_IHT=30,C_IHT=0.9,lamda_IHT=0.1,
                 step_ista=0.0000001,itt_ista=30,C_ista=0.9,lamda_ista=0.1,threshold_ista=2,epsilon=0,lmax=20,r=3,
                 CV_lasso=5,itt_Lasso=100,x_star=0,tol_Lasso=0.001):
        
        super().__init__()
        if r < 3:
            # According to the article r should be greater or equal to 3 
            warnings.warn("The value of 'r' should be greater than or equal to 3. Please set 'r' to a value greater than 3.", UserWarning)
        
        
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
        self.step_ista=step_ista
        self.itt_ista=itt_ista
        self.threshold_ista=threshold_ista
        self.C_ista=C_ista
        self.lamda_ista=lamda_ista
        self.epsilon=epsilon
        self.r=r
        self.lmax=lmax
        self.CV_lasso=CV_lasso
        self.itt_Lasso=itt_Lasso
        self.x_star=x_star
        self.tol_Lasso=tol_Lasso



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
            grad_estimate=Lasso_reg(y,Z,cv=self.CV_lasso,it=self.itt_Lasso,tol=self.tol_Lasso)
        if self.algo=='DLasso':
            grad_estimate=debiased_Lasso(y,Z,delta,cv=self.CV_lasso,it=self.itt_Lasso,tol=self.tol)
        if self.algo=='IHT_Classique':
            grad_estimate=IHT_classique(X=Z,Y=y,s=self.s,step=self.step_IHT,max_iterations=self.itt_IHT)
        if self.algo=='IHT_ad':
            grad_estimate=IHT_ad(X=Z,Y=y,threshold=self.threshold_IHT,C=self.C_IHT,step=self.step_IHT,max_iterations=self.itt_IHT,lamda=self.lamda_IHT)
        if self.algo=='ISTA_ad':
            grad_estimate=ISTA_ad(X=Z,Y=y,threshold=self.threshold_ista,C=self.C_ista,step=self.step_ista,max_iterations=self.itt_ista,lamda=self.lamda_ista)
        return grad_estimate, function_estimate
    

    def step_MD(self):
        '''
        Take step of optimizer
        '''
        # Define the Projection 
        grad_est, f_est = self.GradEstimate()
        self.fd = f_est
        p1=proj.SimplexProjectionExpSort(dimension = self.n, epsilon = self.epsilon)
        p2= proj.SimplexProjectionExpSort(dimension = self.n, epsilon = 0)
        s2=1/self.lmax
        s1=s2*self.epsilon/(1+self.n*self.epsilon)
        acm=AcceleratedMethod(self.f, grad_est, p1, p2, s1, s2, self.r, self.x, 'accelerated descent')
        acm.step()
        x_k_plus_1=acm.x

        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        #self.x = self.Prox(x_k_plus_1) # MD
        self.x=x_k_plus_1

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals, self.x, 'B'

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.x, 'T'
 
        self.t += 1
        return self.function_evals, False, False
    


    def Zoro_MD(self):
        performance_log_ZORO_MD = [[0, self.f(self.x)]]
        cost_x=[[0,np.linalg.norm(self.x-self.x_star)]]
        termination = False
        while termination is False:
            evals_ZORO, solution_ZORO, termination = self.step_MD()
            cost=np.linalg.norm(self.x-self.x_star)
            # save some useful values
            performance_log_ZORO_MD.append( [evals_ZORO,np.mean(self.fd)] )
            cost_x.append([evals_ZORO,cost])
            # print some useful values
            #performance_log_ZORO.append( [evals_ZORO,self.f(solution_ZORO)] )
            self.report( 'Estimated f(x_k): %f x_k-x_star: %f function evals: %d\n' %
            (np.mean(self.fd), cost ,evals_ZORO) )
        return performance_log_ZORO_MD,cost_x
