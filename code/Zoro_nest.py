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
from help_function import ISTA_ad,IHT_ad,IHT_classique,debiased_Lasso,Lasso_reg,True_grad_SparseQuadric,True_grad_square_of_the_difference_support_S,True_grad_norm_with_a_Gaussian_matrix
import projection as proj
from MD import AcceleratedMethod
import warnings
import math


class ZORO_nest(BaseOptimizer):
    '''
    ZORO for black box optimization. 
    '''
    def __init__(self, x0, f, params, algo ,threshold_IHT=2,function_budget=10000,
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
        #self.M=params["M"]


        # Define sampling matrix
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1

        cosamp_params = { "Z": Z,"delta": self.delta, "maxiterations": 10,
                         "tol": 0.5, "sparsity": self.sparsity}
        self.cosamp_params = cosamp_params
        print(f"here {vars(self)}")
    
    def nastrov_step(gradient_t,x_t,y_t_1,lamda_t,gamma_t,alpha=0.05):
        '''
        Nastrov step
        '''
        y_t=y_t_1-alpha*gradient_t
        x_t_plus_1=(1-gamma_t)*x_t+gamma_t*y_t
        lamda_t_plus_1=(1 + math.sqrt(1 + 4 * (lamda_t**2))) / 2
        gamma_t_plus_1=(1-lamda_t)/lamda_t_plus_1
        return x_t_plus_1,y_t,lamda_t_plus_1,gamma_t_plus_1



    """ 
    # Handle the (potential) proximal operator
    def Prox(self, x):
        if self.prox is None:
            return x
        else:
            return self.prox.prox(x, self.step_size)
    """


    
    def GradEstimate(self,x=None):
        '''
        Gradient estimation sub-routine.
        '''
      
        maxiterations = self.cosamp_params["maxiterations"]
        Z = self.cosamp_params["Z"]
        delta = self.cosamp_params["delta"]
        sparsity = self.cosamp_params["sparsity"]
        tol = self.cosamp_params["tol"]
        Z = 2*(np.random.rand(self.num_samples, self.n) > 0.5) - 1
        num_samples = np.size(Z, 0)
        if x is None:
            x = self.x
        f = self.f
        y = np.zeros(num_samples)
        function_estimate = 0
        for i in range(num_samples):
            y_temp = f(x + delta*np.transpose(Z[i,:]))
            y_temp3=f(x - delta*np.transpose(Z[i,:]))
            y_temp2 = f(x)
            #function_estimate += y_temp2
            y[i] = (y_temp - y_temp3)/(2*np.sqrt(num_samples)*delta)
            self.function_evals += 2
        function_estimate= f(x)
        #function_estimate = function_estimate/num_samples
        #Z = Z/np.sqrt(num_samples)
        if self.algo=='CoSaMP':
            grad_estimate = cosamp(Z, y, sparsity, tol, maxiterations)
        if self.algo=='Lasso':
            grad_estimate=Lasso_reg(y,Z,cv=self.CV_lasso,it=self.itt_Lasso,tol=self.tol_Lasso)
        if self.algo=='DLasso':
            grad_estimate=debiased_Lasso(y,Z,delta,cv=self.CV_lasso,it=self.itt_Lasso,tol=self.tol_Lasso)
        if self.algo=='IHT_Classique':
            grad_estimate=IHT_classique(X=Z,Y=y,s=self.s,step=self.step_IHT,max_iterations=self.itt_IHT)
            s_1=np.count_nonzero(grad_estimate)
        if self.algo=='IHT_ad':
            grad_estimate=IHT_ad(X=Z,Y=y,threshold=self.threshold_IHT,C=self.C_IHT,step=self.step_IHT,max_iterations=self.itt_IHT,lamda=self.lamda_IHT)
        if self.algo=='ISTA_ad':
            grad_estimate=ISTA_ad(X=Z,Y=y,threshold=self.threshold_ista,C=self.C_ista,step=self.step_ista,max_iterations=self.itt_ista,lamda=self.lamda_ista)
        return grad_estimate, function_estimate
    

    

    def step(self,y,lamda,gamma):
        '''
        Take step of optimizer
        '''
   
        grad_est, f_est = self.GradEstimate(x=y)
        self.fd = f_est
        true_grad=True_grad_square_of_the_difference_support_S(x=self.x,x_star=self.x_star,s=self.s)
        s_2=np.count_nonzero(true_grad)
        norm_true_Grad=np.linalg.norm(true_grad)
        est_grad_norm=np.linalg.norm(grad_est)
        # Note that if no prox operator was specified then self.prox is the
        # identity mapping.
        self.x,y,lamda,gamma=ZORO_nest.nastrov_step(gradient_t=grad_est,x_t=self.x,y_t_1=y,lamda_t=lamda,gamma_t=gamma)
        #self.x = (self.x -self.step_size*grad_est) # gradient descent 

        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            # if budget is reached return current iterate
            return self.function_evals, self.x, 'B',norm_true_Grad,est_grad_norm,y,lamda,gamma

        if self.function_target is not None:
            if self.reachedFunctionTarget(self.function_target, f_est):
                # if function target is reached terminate
                return self.function_evals, self.x, 'T',norm_true_Grad,est_grad_norm,y,lamda,gamma
 
        self.t += 1
        return self.function_evals, False, False,norm_true_Grad,est_grad_norm,y,lamda,gamma
    


    def Zoro(self):
        performance_log_ZORO = [[0, self.f(self.x)]]
        #cost_x=[[0,np.linalg.norm(self.x-self.x_star)]]
        norm_true_grad=[]
        est_grad_norm_list=[]
        termination = False
        lamda_prev=0
        y_n_prev=self.x
        gamma_prev=1
        while termination is False:
            evals_ZORO, solution_ZORO, termination,norm_true_Grad,est_grad_norm,y_cur,lamda_cur,gamma_cur = self.step(y=y_n_prev,lamda=lamda_prev,gamma=gamma_prev)
            y_n_prev=y_cur
            lamda_prev=lamda_cur
            gamma_prev=gamma_cur

            #cost=np.linalg.norm(self.x-self.x_star)
            
            # save some useful values
            performance_log_ZORO.append( [evals_ZORO,np.mean(self.fd)] )
            #cost_x.append([evals_ZORO,cost])
            norm_true_grad.append([evals_ZORO,norm_true_Grad])
            est_grad_norm_list.append([evals_ZORO,est_grad_norm])
            # print some useful values
            #performance_log_ZORO.append( [evals_ZORO,self.f(solution_ZORO)] )
            self.report( 'Estimated f(x_k): %f norm of the estimated gradient: %f  function evals: %d Norm True grad: %f \n' %
            (np.mean(self.fd),(est_grad_norm_list[-1])[-1] ,evals_ZORO,(norm_true_grad[-1])[-1]) )
        return(performance_log_ZORO,norm_true_grad,est_grad_norm_list)



