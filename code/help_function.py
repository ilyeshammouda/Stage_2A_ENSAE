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

def Lasso_reg(Y,Z,debiased='False'):
        """
        This function gives the solution to the Lasso regression in a multivariate model
        """
        lasso = linear_model.LassoCV(cv=5)
        lasso.fit(Z,Y)
        g = lasso.coef_
        u=lasso.intercept_
        if debiased=='False':
                return(g)
        else:
                return(g,u)


def debiased_Lasso(Y,Z,delta):
        n=Z.shape[0]
        ones=np.ones(n)
        g_t,u=Lasso_reg(Y,Z,debiased='True')
        y_tilde=Y/delta
        g_tilde=g_t+((1/n)*Z.T@(y_tilde-u*ones))
        return(g_tilde)



def s_grands_comp(vecteur, s):
        indices_tries = np.argsort(vecteur)[::-1]
        valeurs_triees = vecteur[indices_tries]
        vecteur_s = np.zeros(len(vecteur))
        for i in range(s):
            vecteur_s[indices_tries[i]] = valeurs_triees[i]
        return vecteur_s


def IHT_classique(X, Y,s,step=0.00000001,max_iterations=30):
        n,m=X.shape
        Z,beta_hat=np.zeros(m),np.ones(m)     
        for i in range(max_iterations):
            Z=beta_hat+(step*(X.T)@(Y-X@beta_hat))
            beta_hat=s_grands_comp(Z, s)
        return beta_hat


def HardThreshold(x,lamda):
        return x*(np.abs(x)>=lamda)


def SoftThreshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
def IHT_ad(X, Y,threshold ,C=0.9,step=0.0000001,max_iterations=30,lamda=0.1):
        n,m=X.shape
        Z,beta_hat=np.zeros(m),np.ones(m)  
        for i in range(max_iterations):
            Z=beta_hat+(step*(X.T)@(Y-X@beta_hat))
            beta_hat=HardThreshold(Z, lamda)
            while lamda > threshold:
                lamda*=C
        return beta_hat


def ISTA_ad(X, Y,threshold ,C=0.9,step=0.0000001,max_iterations=30,lamda=0.1):
        n,m=X.shape
        Z,beta_hat=np.zeros(m),np.ones(m)  
        for i in range(max_iterations):
            Z=beta_hat+(step*(X.T)@(Y-X@beta_hat))
            beta_hat=SoftThreshold(Z, lamda)
            while lamda > threshold:
                lamda*=C
        return beta_hat
    


