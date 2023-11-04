""" 
This code defines methods to solve the MD probeleme. 
The original code can be found in this Git repo: "https://github.com/walidk/AcceleratedMirrorDescent". 
The methodes implemented in this code are those defined in the article: "https://papers.nips.cc/paper_files/paper/2015/file/f60bb6bb4c96d4df93c51bd69dcc15a0-Paper.pdf".

"""




import projection as proj
import numpy as np

class AcceleratedMethod(object):
    def __init__(self, f, gradf, p1, p2, s1, s2, r, x0, name, alpha = 2):
    # f: objective function
    # gradf: gradient of the objective function
    # p1 and p2 are projections
    # p1.project(x, g) returns
    #   in the unconstrained case: nablaPsi(nablaPsiInverse(x) - g)
    #   in the simplex constrained case: \phi(\phi^{-1}(x_i) - g_i + \nu)_+ for an optimal \nu
    # s1, s2 are the step sizes for the primal and dual step
    # r is a scalar parameter in the ODE (needs to be strictly larger than 2)
        self.k = 1
        # projection operators
        self.p1 = p1
        self.p2 = p2
        # step sizes
        self.s1 = s1
        self.s2 = s2
        # Energy rate and averaging rate
        if(r < alpha):
            raise Exception('r needs to be larger than alpha (default 2)')
        self.alpha = alpha
        self.r = r
        # Primal variable initialization
        self.x0 = x0
        self.z0 = x0
        self.xtilde = x0
        self.x = x0
        self.z = x0
        self.xprev = x0
        self.zprev = x0
        # objective and gradient oracles
        self.gradf = gradf
        self.f = f
        self.name = name

    def __str__(self):
        return 'Accelerated method with s1={}, s2={}, p1={}, p2={}, r={}'.format(s1, s2, p1, p2, r)
    def restart(self):
        # print('restarted at iteration {}'.format(self.k))
        self.z = self.xprev
        self.x = self.xprev
        self.k = 1
    def reset(self):
        self.x = self.x0
        self.z = self.x0
        self.xtilde = self.x0
        self.xprev = self.x0
        self.zprev = self.x0
        self.k = 1
    def step(self):
        
        r = self.r
        k = self.k
        x = self.x
        z = self.z
        g = self.gradf

        xtilde = self.p1.project(x, self.s1 * g)
        self.xprev = self.x
        self.x = xtilde  # Update x with xtilde
        self.xtilde = xtilde  # Save xtilde for the next iteration
        #self.z = ztilde
        self.k = k + 1
        
        """
        r = self.r
        k = self.k
        x = self.x
        z = self.z
        g = self.gradf
        xtilde = self.p1.project(x, self.s1*g)
        xtilde = x
        ztilde = self.p2.project(z, k**(self.alpha - 1)*self.s2/r*g)
        xp = (xtilde + r/k**(self.alpha - 1)*ztilde)/(1+r/k**(self.alpha - 1))
        self.xprev = self.x
        self.zprev = self.z
        self.xtilde = xtilde
        self.x = xp
        self.k = k+1
        """