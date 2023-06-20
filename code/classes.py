#import packages
import numpy as np
from sklearn import linear_model

class tools:
    def Rademacher_matrix(d,n):
        """
        This fucntion generates a Rademacher matrix
        """
        return np.random.choice([-1, 1], size=(d,n))



    def Rademacher_matrix_concatenated(d,n):
        """
        This function generates a Rademacher matrix and add a line of ones 
        """
        Z=tools.Rademacher_matrix(d,n)
        Last_line_of_ones = np.ones((1, Z.shape[1]))
        return (Z,np.concatenate((Z, Last_line_of_ones), axis=0))
    



class algos:
    def Lasso_reg(Y_tilde,Z,lamda=0.1):
        """
        This function gives the solution to the Lasso regression in a multivariate model
        """
        lasso = linear_model.Lasso(lamda)
        lasso.fit(Z,Y_tilde )
        g = lasso.coef_
        u=lasso.intercept_
        return(g,u)


    def GradiantEstimate(y_t_vect,d,n,delta,lamda=0.1):
        """ 
        This function corresponds to the pseudo algorithme 1 defined in the paper
        """
        Z=tools.Rademacher_matrix(n,d)
        y_tilde=y_t_vect/delta
        (g,u)=algos.Lasso_reg(Y_tilde=y_tilde,Z=Z,lamda=lamda)
        return(g,u)
    

        
class Tests:
    # Functions that will be used for testing 
    def f_test(x_t,delta,d,noise=1):
        Z=tools.Rademacher_matrix(d,1)
        return(np.linalg.norm(x_t+delta*Z)+noise*np.random.normal(0,1,1))
    def vect_f_test(x_t, delta, d, n):
        y_t_vecteur = np.zeros(n)
        for i in range(n):
            y_t_vecteur[i] = Tests.f_test(x_t, delta, d)
        y_t = Tests.f_test(x_t=x_t, delta=0, d=d, noise=0)
        return y_t_vecteur, y_t
    