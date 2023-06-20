from classes import Tests,tools,algos
import numpy as np



#fix parameters

d=100
n=50
delta=0.5
lamda=0.1
x_t=np.random.binomial(1, 1/2,size=(d,))

#simulating a vector of [y_i] i in [1,n] where y_i is are noisy obeservations of the true y_t
y_t_vecteur,y_t=Tests.vect_f_test(x_t,delta,d,n)
print(f"The true value of y_t: {y_t}")
print(f"The neighbourhood around y_t \n {y_t_vecteur} ")

# Performing the estimation of the gradiant of a given function say f in the point x_t
g_t,u_t=algos.GradiantEstimate(y_t_vecteur,d,n,delta)
print(f" The lasso estimator of g_t and u_t are:\n  g_t= {g_t} \n u_t= {u_t} ")
