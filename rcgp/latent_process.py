import torch as tc
import numpy as np
from scipy.special import binom
from scipy.linalg import solve_continuous_lyapunov

from .utils import eye

class MaternProcess():

    def __init__(self, p : int, magnitude : tc.Tensor, lengthscale : tc.Tensor):

        #sizes
        self.p = p #e.g., x, x', x'' has p=2
        
        #matern process constants
        self.magnitude = magnitude
        self.lengthscale = lengthscale
        self.nu = tc.tensor(1/2 + self.p)
        self.lamb = tc.sqrt(2 * self.nu) / self.lengthscale
        self.spectral_density = self.magnitude * 2 * tc.pi **(1/2) * tc.lgamma(self.nu + 1/2).exp() * (2 * self.nu)**(self.nu) / tc.lgamma(self.nu).exp() / (self.lengthscale ** (2 * self.nu))

        self.Ft = eye(p+1, k=1)
        self.Ft[-1, :] = tc.stack([- binom(p+1, i) * self.lamb**(p+1 - i) for i in range(p+1)])
        self.Lt = eye(1, p+1, k=p).T

        self.L_np = self.Lt.cpu().detach().numpy()
        self.F_np = self.Ft.cpu().detach().numpy()
        self.sol_lyapunov_np = solve_continuous_lyapunov(self.F_np, -self.L_np @ self.L_np.T)

        self.Sig0 = self.spectral_density * tc.tensor(self.sol_lyapunov_np) 

        return
    
    def forward(self, dt):
        A = tc.linalg.matrix_exp(dt * self.Ft)

        Sig = self.Sig0 - A @ self.Sig0 @ A.T

        return A, Sig