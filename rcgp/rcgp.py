import torch as tc
from torch import nn
import torch.nn.functional as F

from typing import Tuple
from .utils import eye
from .latent_process import MaternProcess


def IMQ_and_gradient(Y: tc.Tensor, m: tc.Tensor, beta: tc.Tensor, c: float):
    assert Y.shape == m.shape, f"Y has shape {Y.shape} and m has shape {m.shape}. They must be of the same size."
    
    # Precompute shared intermediate terms
    diff = (Y - m) / c                      # Element-wise division
    diff_squared = diff.pow(2)              # Square of the difference
    one_plus_diff_squared = 1 + diff_squared  # (1 + ((Y - m) / c)^2)
    
    # Compute reciprocal square root for the IMQ term
    rsqrt_term = one_plus_diff_squared.rsqrt()  # 1 / sqrt(1 + diff^2)
    
    # IMQ term
    IMQ = beta * rsqrt_term
    
    # Gradient-like term
    gradient_term = -beta * diff * rsqrt_term.pow(3)  # -beta * diff / (1 + diff^2)^(3/2)
    
    return IMQ, gradient_term


class TemporalRCGP(nn.Module):
    """
    Temporal Robust and Conjugate Gaussian Process.
    """
    def __init__(self, 
                 ts : tc.Tensor,
                 Ys : tc.Tensor, p : int = 1,
                 raw_var_y_init : float = 1.0,
                 raw_temporal_lengthscale_init : float = 1.0,
                 raw_temporal_magnitude_init : float = 1.0,
                 ):
       
        super().__init__()
        
        #Initial checks
        (self._n_t, self.d) = self.initialize(p=p, ts=ts, Ys=Ys)
        
        self._ts = ts.to(tc.float32) #original ts (without padding)
        self._Ys = Ys.to(tc.float32) #original Ys (without padding)

        #Padding filtering time array and data array
        (self.n_t, self.Ys, self.ts) = self.padding(ts=self._ts, Ys=self._Ys)

        self.__prior_mean_funcs = ["constant", "m_pred"]

        self.__fixed_params = {"p" : p, #temporal Matern kernel (nu_{temporal} = p + 1/2)
                               "robust" : False, #Not automatically robust!
                               "prior_mean" : "constant",
                               "beta" : None, #If None, will be sqrt(var_y / 2)
                               "is_beta_fixed" : False,
                               "c" : 1., #If None and is_c_fixed=True, will be c_factor * sqrt(var_y). 
                               "c_factor" : 1.,
                               "is_c_fixed": True, #When false, c is adaptive and = 2 * predicted std
                               }

        self._raw_var_y = nn.Parameter(tc.tensor(raw_var_y_init, dtype=tc.float32))
        self._raw_temporal_lengthscale = nn.Parameter(tc.tensor(raw_temporal_lengthscale_init, dtype=tc.float32))
        self._raw_temporal_magnitude = nn.Parameter(tc.tensor(raw_temporal_magnitude_init, dtype=tc.float32))


        self.H = eye(1, (1+self.__fixed_params["p"]), k=0).to(tc.float32) #measurement matrix -> temporal only case
        self.latent_size = p + 1
        return
    

    @property
    def fixed_params(self):
        return self.__fixed_params

    @property
    def var_y(self):
        return F.softplus(self._raw_var_y)
    
    @var_y.setter
    def var_y(self, value):
        """
        Setter for the variance in y.
        """
        if value <= 0:
            raise ValueError("Parameter must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        self._raw_var_y.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    @property
    def beta(self):
        if self.__fixed_params["is_beta_fixed"]:

            if isinstance(self.__fixed_params["beta"], (float, int)):
                return tc.tensor(self.__fixed_params["beta"], dtype=tc.float32)
            
            elif isinstance(self.__fixed_params["beta"], tc.Tensor):
                return self.__fixed_params["beta"]
            else:
                raise ValueError(f"{self.__fixed_params["beta"]} is an invalid data type for beta.")
        else:
            return tc.sqrt(self.var_y / 2)#.clone().detach()
    
    @beta.setter
    def beta(self, value : float):
        if isinstance(value, tc.Tensor):
            self.__fixed_params["beta"] = value

        elif isinstance(value, (float, int)):
            self.__fixed_params["beta"] = tc.tensor(value, dtype=tc.float32)

        elif value is None: #If we set beta = None, we want to revert back to adaptive beta. 
            self.__fixed_params["beta"] = None
            self.__fixed_params["is_beta_fixed"] = False
            return
        else:
            raise ValueError(f"{value} is not a valid value for parameter beta.")
        
        self.__fixed_params["is_beta_fixed"] = True
        return
    
    @property
    def c(self):
        if isinstance(self.__fixed_params["c"], (float, int)):
            return tc.tensor(self.__fixed_params["c"], dtype=tc.float32)
        
        elif isinstance(self.__fixed_params["c"], tc.Tensor):
            return self.__fixed_params["c"]
        else:
            return self.__fixed_params["c_factor"] * tc.sqrt(self.var_y)#.clone().detach()
    
    @c.setter
    def c(self, value : float):
        if isinstance(value, tc.Tensor):
            self.__fixed_params["c"] = value
            
        elif isinstance(value, (float, int)):
            self.__fixed_params["c"] = tc.tensor(value, dtype=tc.float32)

        elif value is None:
            self.__fixed_params["c"] = None

        else:
            raise ValueError(f"{value} is not a valid value for parameter c.")
        
        self.__fixed_params["is_c_fixed"] = True
        return
    
    def adaptive_c(self):
        self.__fixed_params["is_c_fixed"] = False
        return

    @property
    def temporal_lengthscale(self):
        return F.softplus(self._raw_temporal_lengthscale)
    
    @temporal_lengthscale.setter
    def temporal_lengthscale(self, value):
        """
        Setter for the temporal lengthscale.
        """
        if value <= 0:
            raise ValueError("Temporal lengthscale must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        if isinstance(value, tc.Tensor):
            self._raw_temporal_lengthscale.data = tc.log(tc.exp(value) - 1)
        else:
            self._raw_temporal_lengthscale.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    @property
    def temporal_magnitude(self):
        return F.softplus(self._raw_temporal_magnitude)
    
    @temporal_magnitude.setter
    def temporal_magnitude(self, value):
        """
        Setter for the temporal magnitude.
        """
        if value <= 0:
            raise ValueError("Temporal magnitude must be positive.")
        # Reverse transform: raw_parameter = log(exp(value) - 1)
        if isinstance(value, tc.Tensor):
            self._raw_temporal_magnitude.data = tc.log(tc.exp(value) - 1)
        else:
            self._raw_temporal_magnitude.data = tc.log(tc.exp(tc.tensor(value, dtype=tc.float32)) - 1)

    
    def initialize(self, p, ts, Ys) -> Tuple[int, int, int]:

        #checks on p
        assert type(p) == int and p > 0, "p must be an integer greater than 0."

        #checks on ts
        assert ts.ndim == 2 and ts.shape[1] == 1, "ts must be two-dimensional, with n_t rows and one column."
        if tc.isnan(ts).any(): raise ValueError("time values contain NaN values.")
        if tc.isinf(ts).any(): raise ValueError("time values contain Inf values.")
        if not tc.all(ts[1:] - ts[:-1] > 0): ValueError("time values are not strictly increasing.")


        assert Ys.ndim == 2 and Ys.shape[0] == len(ts) and Ys.shape[1] == 1, "Temporal data must have shape (n_t, 1)."

        n_t = ts.shape[0]
        d = 1
        return n_t, d
 

    def padding(self, ts : tc.Tensor, Ys : tc.Tensor):

        t0 = (ts[0] - (ts[1] - ts[0])).unsqueeze(0)
        tf = (ts[-1] + (ts[-1] - ts[-2])).unsqueeze(0)

        ts_filtering = tc.concatenate([t0, ts, tf])
        n_t_filtering = ts_filtering.shape[0]

        nan_row = tc.full((1, 1), float('nan'))
        Ys_filtering = tc.cat([nan_row, Ys, nan_row], dim=0)

        return n_t_filtering, Ys_filtering.to(tc.float32), ts_filtering.to(tc.float32)
    

    def v(self, Y : tc.Tensor, weights : tc.Tensor, partial_weights : tc.Tensor):

        if not self.__fixed_params["robust"]:
            R_inv = self.R_inv(Y=Y, weights=weights, partial_weights=partial_weights)
            return R_inv @ Y
        else:
            return (2/self.var_y**2) * (weights**2 * Y - 2 * self.var_y * weights * partial_weights)

    def R_inv(self, Y : tc.Tensor, weights : tc.Tensor, partial_weights : tc.Tensor):

        if not self.__fixed_params["robust"]:
            return 1/self.var_y * tc.eye(Y.shape[1])
        else:
            return (2/self.var_y**2) * weights.squeeze()**2


    def activate_robustness(self, func : str):
        if func in self.__prior_mean_funcs:
            self.__fixed_params["prior_mean"] = func
            self.__fixed_params["robust"] = True
        else:
            raise ValueError(f"{func} is not in {self.__prior_mean_funcs}.")

        return
    
    def deactivate_robustness(self):
        self.__fixed_params["robust"] = False
        return
    
    def set_prior_mean(self, func : str):
        self.__fixed_params["prior_mean"] = func

        
    def prior_mean(self, Y : tc.Tensor, m_prior : tc.Tensor):

        if self.__fixed_params["prior_mean"] == 'constant':
            mean = self.Ys.nanmean() * tc.ones_like(Y) #nan mean because we might encounter NaNs in the data (we actually buffer with NaNs)
            return mean
        
        elif self.__fixed_params["prior_mean"] == 'm_pred':
            return m_prior

        else:
            return m_prior

        
    def compute_weights(self, Y : tc.Tensor, m_prior : tc.Tensor, P_prior : tc.Tensor):

        if not self.__fixed_params["robust"]:
            return tc.sqrt(self.var_y / 2).reshape(1,1).clone().detach(), tc.tensor([[0.0]], dtype=tc.float32)
        
        else:
            m = self.prior_mean(Y=Y, m_prior=m_prior)

            if self.__fixed_params["is_c_fixed"]:
                c = self.c
            else:
                with tc.no_grad():
                    c = tc.sqrt(P_prior + self.var_y - self.var_y * P_prior).squeeze() #.clone().detach()


            weights, partial_y_weights = IMQ_and_gradient(Y=Y, m=m, beta=self.beta, c=c)

            return weights, partial_y_weights
    

    def get_predict_step_matrices(self, dt : float, temporal_kernel : MaternProcess):
        At, Sigt = temporal_kernel.forward(dt)
        A = At
        Sig = Sigt

        return A.to(tc.float32), Sig.to(tc.float32)


    def predict_step(self, A : tc.Tensor, Sig : tc.Tensor, m : tc.Tensor, P : tc.Tensor):
        m_pred = A @ m  # Predict state
        A_P = A @ P
        P_pred = A_P @ A.T + Sig  # Predict covariance
        return m_pred.to(tc.float32), P_pred.to(tc.float32)
    
    def update_step(self, m_pred : tc.Tensor, P_pred : tc.Tensor, Y : tc.Tensor):

        # Precompute R_inv_k and R_k
        f_hat = self.H @ m_pred #Prediction of mean
        H_P_pred = self.H @ P_pred  # (1 x n) vector


        weights, partial_weights = self.compute_weights(Y=Y, m_prior=f_hat, P_prior=H_P_pred @ self.H.T) #Weights Computation

        R_inv_k = self.R_inv(Y=Y, weights=weights, partial_weights=partial_weights).squeeze()
        R_k = 1.0 / R_inv_k
        
        # Compute intermediate terms
        S_k = R_k + H_P_pred @ self.H.T  # Scalar
        
        # Kalman gain (n x 1) vector
        K_k = (H_P_pred.T / S_k)  # Simplified scalar division
        
        # Innovation term (scalar)
        v_k = self.v(Y=Y, weights=weights, partial_weights=partial_weights)  # Scalar
        innovation = R_k * v_k - f_hat  # Scalar
        
        # Updated mean
        m_updated = m_pred + K_k * innovation  # (n x 1)
        
        # Updated covariance
        P_updated = P_pred - (K_k @ H_P_pred)  # (n x n)
    
        return m_updated, P_updated, weights
    
    def filtsmooth(self, temporal_kernel : MaternProcess, m0 : tc.Tensor, P0 : tc.Tensor, smoothing=False):

        #Where we store the state, covariance estimates m_{k|k}, P_{k|k}
        ms = tc.empty(size=(self.n_t, self.latent_size, 1), dtype=tc.float32) #n_t time steps where n_t is len(ts) + 2 (padded). latent size depends on number of spatial dimensions and temporal derivatives
        Ps = tc.empty(size=(self.n_t, self.latent_size, self.latent_size), dtype=tc.float32) #same here. 

        #Container for the weights. Choosing lists because might have to autodiff through weights and inplace assignment is no good for autodiff.
        W0 = tc.tensor(0.0, dtype=tc.float32).reshape(1,1)
        Ws = [W0]
        
        #Where we store the one-step predictions/predictives  p(z_k | y_{1:k-1})
        m_preds, P_preds = [m0], [P0]

        #Initialize the 0th step. No data observed on 0th step. We always do look-ahead predict and update (i.e. from k, predict k+1 and update with data at k+1)
        m, P = m0, P0 #m, P are the "running" state and covariance matrix. They get overwritten. 

        ms[0], Ps[0] = m0.clone().detach(), P0.clone().detach()

        dt = (self.ts[1] - self.ts[0]).item()

        A, Sig = self.get_predict_step_matrices(dt=dt, temporal_kernel=temporal_kernel) #Get the transition matrix and the noise matrix

        #Filtering
        for k in range(self.n_t - 1):
            #Compute time difference from now to next step
            dt = (self.ts[k+1] - self.ts[k]).item()

            #--------------------Predict Step------------------------
            m_pred, P_pred = self.predict_step(A=A, Sig=Sig, m=m, P=P) #Get state,covariance estimate of z_k for y_{k+1}
            
            m_preds.append(m_pred)
            P_preds.append(P_pred)

            #---------------------Update Step------------------------
            if tc.isnan(self.Ys[k+1]).any(): #If data is incomplete we don't perform update step. 
                m = m_pred
                P = P_pred

                Ws.append(W0.clone().detach()) #Weights are zero because we don't want to include in calculation a value of 0.
                ms[k+1] = m_pred.clone().detach() #We store the predictive mean m_{k+1|k} as the mean estimate m_{k+1|k+1}
                Ps[k+1] = P_pred.clone().detach() #We store the predictive step cov P_{k+1|k} as the cov estimate P_{k+1|k+1}

            else: #Update with data at k+1 when there is no missing data
                m_updated, P_updated, weights = self.update_step(m_pred=m_pred, P_pred=P_pred, Y=self.Ys[[k+1]])

                m = m_updated
                P = P_updated

                ms[k+1] = m_updated.clone().detach()
                Ps[k+1] = P_updated.clone().detach()
                Ws.append(weights.clone().detach()) 

        #Stacking
        Ws = tc.concatenate(Ws) 
        m_preds = tc.stack(m_preds)
        P_preds = tc.stack(P_preds)

        if smoothing:
            #RTS Smoothing
            for k in range(self.n_t - 2, -1, -1):  
                C = Ps[k] @ A.T @ tc.linalg.inv(P_preds[k+1].clone().detach())

                ms[k] = ms[k] + C @ (ms[k+1] - m_preds[k+1].clone().detach())
                Ps[k] = Ps[k] + C @ (Ps[k+1] - P_preds[k+1].clone().detach()) @ C.T
            
        return ms, Ps, m_preds, P_preds, Ws
    

    def forward(self, smoothing=True):
        
        #Instantiate the temporal kernel
        temporal_kernel = MaternProcess(p=self.__fixed_params["p"],
                                        lengthscale=self.temporal_lengthscale,
                                        magnitude=self.temporal_magnitude,
                                        )

        #If no spatial 
        
        m0=tc.zeros(size=(self.latent_size, 1), dtype=tc.float32)
        P0=temporal_kernel.Sig0.to(tc.float32)

        ms, Ps, m_preds, P_preds, Ws = self.filtsmooth(m0=m0, P0=P0, temporal_kernel=temporal_kernel, smoothing=smoothing)

        preds_filt = (self.H @ m_preds)[1:-1]
        preds_filt = preds_filt.squeeze(-1)
            
        covs_filt = (self.H @ P_preds @ self.H.T)[1:-1]

        preds_smooth = (self.H @ ms)[1:-1].squeeze(-1)
        covs_smooth = (self.H @ Ps @ self.H.T)[1:-1]
        stds_smooth = tc.sqrt(tc.diagonal(covs_smooth, dim1=1, dim2=2)).squeeze()

        R = self.var_y * tc.eye(1)
        Ws_norm = Ws[1:-1] / Ws.sum()
        eff = (Ws.clone().detach() / self.beta.clone().detach())[1:-1]

        return (preds_smooth, stds_smooth, eff), (preds_filt, covs_filt, R, Ws_norm), (ms, Ps)
